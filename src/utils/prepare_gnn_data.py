import networkx
import torch
import dgl
from dgl.data.utils import save_graphs, save_info, load_graphs, load_info
import os
import itertools
import logging
import numpy as np
from pathlib import Path

from src.utils.corpus_readers import ProcessedIterExtended, get_dataloader
from src.utils.prepare_laat_data import DataReader, Dataset
from src.utils.config import PROJ_FOLDER


os.environ['DGLBACKEND'] = 'pytorch'

logger = logging.getLogger(__name__)


class GNNDataReader(DataReader):
    """
    Read datafile(s) from both text .csv files and pre-processed lined_data dataset files for CUI inputs and prepare
    data for GNNDataset class.

    1) Each sample will be returned as list of unique CUIs
    2) MLB for each sample labels
    3) Calculate stats for each partition
    """

    def __init__(self,
                 data_dir=f"{PROJ_FOLDER / 'data' / 'mimic3'}",
                 version="50",
                 input_type="umls",
                 prune_cui=True,
                 cui_prune_file="50_cuis_to_discard_snomedcase4.pickle",
                 vocab_fn="processed_full_text_pruned.json",
                 max_seq_length=None,
                 doc_iterator=None,
                 umls_iterator=None,
                 second_txt_vocab_fn=None,
                 threshold=0.7):
        super().__init__(data_dir=data_dir,
                         version=version,
                         input_type=input_type,
                         prune_cui=prune_cui,
                         cui_prune_file=cui_prune_file,
                         vocab_fn=vocab_fn,
                         max_seq_length=max_seq_length,
                         doc_iterator=doc_iterator,
                         umls_iterator=umls_iterator,
                         second_txt_vocab_fn=second_txt_vocab_fn,
                         threshold=threshold)
        self.doc2len = dict()

    def _fit_transform(self, split):
        # only need CUIs for now
        # return doc ID, iput CUIs, only unique CUIs, doc len == num of unique CUIs that have been flattened
        if self.input_type == "text":
            raise NotImplementedError(f"Invalid input_type: only CUI input is supported!")
        elif "umls" in self.input_type:
            umls_doc_iter = self.umls_doc_iterator(self.umls_doc_split_path[split],
                                                   threshold=self.confidence_threshold,
                                                   pruned=self.prune_cui,
                                                   discard_cuis_file=self.cui_prune_file)
            text_id_iter = self.doc_iterator(self.doc_split_path[split], slice_pos=0)
            text_lab_iter = self.doc_iterator(self.doc_split_path[split], slice_pos=3)
            for doc_id, (umls_data), doc_labels in zip(text_id_iter, umls_doc_iter, text_lab_iter):
                u_id, u_sents, u_len = umls_data
                tokens = itertools.chain.from_iterable(u_sents)
                # tokens are CUIs tokens, flattened
                unique_tokens_only = set(tokens)  # each CUI token will be a node
                input_ids = self.featurizer.convert_tokens_to_features(unique_tokens_only, max_seq_length=None)
                # input ids correspond to row idx in .npy embedding files and vocab .json files
                self.doc2labels[doc_id] = doc_labels

                assert len(unique_tokens_only) == len(input_ids), "input ids not of the same length as input tokens!"

                try:
                    self.doc2len[split][doc_id] = len(unique_tokens_only)
                except KeyError:
                    self.doc2len[split] = dict()
                    self.doc2len[split][doc_id] = len(unique_tokens_only)

                yield doc_id, input_ids, list(unique_tokens_only), self.mlb.transform([doc_labels]), len(
                    unique_tokens_only)
        else:
            raise NotImplementedError(f"Invalid input_type option!")

    def get_dataset_stats(self, split):
        if self.split_stats[split].get('mean') is not None:
            return self.split_stats[split]

        if self.input_type == "text":
            raise NotImplementedError(f"Invalid input_type: only CUI input is supported!")
        elif "umls" in self.input_type:
            try:
                doc_lens = list(self.doc2len[split].values())
            except KeyError:
                _ = list(self._fit_transform(split))
                doc_lens = list(self.doc2len[split].values())
        else:
            raise NotImplementedError(f"Invalid input_type option!")

        self.split_stats[split]['min'] = np.min(doc_lens)
        self.split_stats[split]['max'] = np.max(doc_lens)
        self.split_stats[split]['mean'] = np.mean(doc_lens)

        return self.split_stats[split]

    # def get_dataset(self, split): --> unchanged from prepare_laat_data.DataReader.get_dataset(self, split)


class GNNDataset(dgl.data.DGLDataset):

    def __init__(self,
                 dataset,
                 mlb,
                 name="train",
                 embedding_type="snomedcase4",
                 version="full",
                 mode="base",  # graph building mode, base vs <name_of_experiment>
                 self_loop=True,
                 raw_dir=f"{PROJ_FOLDER / 'data'}",
                 save_dir=None,  # save_path = os.path.join(save_dir, self.name)
                 force_reload=False,
                 verbose=False,
                 transform=None,
                 ):

        self._name = name  # MIMIC-III-CUI train/dev/test partition
        self.ds_name = "mimic3_cui"
        self.embedding_type = embedding_type if embedding_type is not None else "snomedcase4"
        self.version = f"{version}"  # sacred commandline parser parses num as int, version needs to be str
        self.mode = mode if mode is not None else "base"

        self.data = dataset  # DataReader.get_dataset('<split: train/dev/test>')
        # self.id2label = {k: v for k, v in enumerate(self.mlb.classes_)}
        self.mlb = mlb  # has already been fit with all label classes

        self.cui2tui = dict()  # mapping cui to tui from semantic_info.csv <--\t separated, col 2
        self.cui2sg = dict()  # mapping cui to semantic group from semantic_info.csv <-- \t separated col 4
        if "kg_rel" in mode:
            self.cui2cui = dict()  # mapping src_cui to dst_cui where a non inverse_isa rel exists between them

        self.self_loop = self_loop if self_loop is not None else True
        self.graphs = []
        self.labels = []
        self.gnidx_cuis = []  # this will be saved with info_dict

        # label dict mapping idx to labels?
        self.glabel_dict = {}  # mapping of mlb.classes_ idx to original labels, classes_ global across train/test/dev
        self._partition_label_dict = {}  # mapping idx to actual label classe in this partition only
        self.nlabel_dict = {}
        self.elabel_dict = {}
        self.ndegree_dict = {}
        self.cc_dict = {}  # mapping graph idx to subgraphs

        # global num
        self.N = len(self.data)  # total graphs number
        self.n = 0  # total nodes number
        self.m = 0  # total edges number
        self.cc = 0  # total number of subgraphs

        # global num of classes
        self.gclasses = 0  # number of classes in the whole dataset
        self.pclasses = 0  # number of classes in this partition
        self.nclasses = 0  # number of node classes
        self.eclasses = 0
        self.dim_nfeats = 0

        # flags
        self.nattrs_flag = False
        self.nlabels_flag = False

        super().__init__(
            name=name,
            hash_key=(name, embedding_type, version, mode, self_loop),
            raw_dir=raw_dir,
            save_dir=save_dir if save_dir is not None else f"{Path(raw_dir) / 'gnn_data' / str(version)}",
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    @property
    def raw_path(self):
        return os.path.join(self.raw_dir, "gnn_data", self.version)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, index):
        """Get the idx-th sample.
                Parameters
                ---------
                index : int
                    The sample index.
                Returns
                -------
                (:class:`dgl.Graph`, Tensor)
                    The graph and its label.
                """
        if self._transform is None:
            g = self.graphs[index]
        else:
            g = self._transform(self.graphs[index])
        return g, self.labels[index], self.gnidx_cuis[index]
        # doc_id, input_ids, labels_bin, doc_len = self.data[index]
        # return input_ids, labels_bin

    def _sem_file_path(self):
        return os.path.join(
            self.raw_dir,
            "umls",
            "semantic_info.csv"
        )

    def _kg_file_path(self):
        cui_version_to_dir_mapping = {"snomedcase4": "case4",
                                         "snomedbase": "umls",
                                         "snomednoex": "umls_noex"}
        cui_embedding_dir = cui_version_to_dir_mapping.get(self.embedding_type, "case4")
        return os.path.join(self.raw_dir,
                            cui_embedding_dir,
                            "all.txt"
                            )

    def _emb_file_path(self):
        """
        Options for self.embedding_type: snomedcase4, snomedbase, snomednoex, umls(W2V)
        """
        return os.path.join(
            self.raw_dir,
            "linked_data",
            "model",
            f"processed_full_{self.embedding_type}_pruned.npy"
        )

    def _build_cui2cui_kg_rel_dict(self):
        kg_rel_file = self._kg_file_path()
        kg_rel_iter = ProcessedIterExtended(kg_rel_file, header=False, delimiter="\t")
        if self.verbose:
            print(f"Processing KG relations from {kg_rel_file}...")
        for row in kg_rel_iter:
            # mapping a cui to other cui with any relation in kg including reciprocal ones (inverse_isa)
            # one cui may be connected to many cui's through different relations
            src_cui, rel, dst_cui = row
            try:
                self.cui2cui[src_cui].append(dst_cui)
            except KeyError:
                self.cui2cui[src_cui] = [dst_cui]

    def _build_graph_edges(self, n_nodes, input_tokens):
        # create dgl graph for each doc
        g = dgl.graph(([], []))
        g.add_nodes(n_nodes)
        m_edges = 0
        if self.self_loop:
            # product includes self element
            groupby_iterable = itertools.product(range(n_nodes), repeat=2)
        else:
            # permute doesn't include self element
            groupby_iterable = itertools.permutations(range(n_nodes), r=2)

        if "base" in self.mode:
            if "kg_rel" in self.mode:
                # connect cuis only if they have KG relations + self edge
                for src_dst_pair in groupby_iterable:
                    src_idx, dst_idx = src_dst_pair
                    src_cui, dst_cui = input_tokens[src_idx], input_tokens[dst_idx]
                    if not g.has_edges_between(src_idx, dst_idx):
                        # only add edges if it's not there already, dgl doesn't check for this by default
                        if dst_cui in self.cui2cui.get(src_cui, []) or (src_idx == dst_idx):
                            g.add_edges(src_idx, dst_idx)
                            m_edges += 1
            else:
                # connect edges only based on common TUIs
                for src_dst_pair in groupby_iterable:
                    src_idx, dst_idx = src_dst_pair
                    src_cui, dst_cui = input_tokens[src_idx], input_tokens[dst_idx]
                    if (self.cui2tui[src_cui] == self.cui2tui[dst_cui]) and not g.has_edges_between(src_idx, dst_idx):
                        g.add_edges(src_idx, dst_idx)
                        m_edges += 1

        elif "combined_tui_kg_rel" in self.mode:
            """
            same as kg_rel
            and in addition, if src_idx in_degrees > 0 or dst_idx out_degrees > 0,
            also connect src and dst sharing same TUI.
            The idea is to have a more connected graph representing the document and fewer disconnected components
            """
            for src_dst_pair in groupby_iterable:
                src_idx, dst_idx = src_dst_pair
                src_cui, dst_cui = input_tokens[src_idx], input_tokens[dst_idx]
                if not g.has_edges_between(src_idx, dst_idx):
                    # only add edges if it's not there already, dgl doesn't check for this by default
                    if dst_cui in self.cui2cui.get(src_cui, []) or (src_idx == dst_idx):
                        g.add_edges(src_idx, dst_idx)
                        m_edges += 1
                    elif self.cui2tui[src_cui] == self.cui2tui[dst_cui]:
                        if g.in_degrees(src_idx) > 0 or g.out_degrees(dst_idx) > 0:
                            g.add_edges(src_idx, dst_idx)
                            m_edges += 1

        elif "combined_kg_ehr" in self.mode:
            """
            connect src_cui to dst cui only under the following criteria:
            1) src_cui in Diagnostic SG/TUI AND dst_cui in PROC SG AND their P(dst_cui|src_cui) > threshold e.g.
            0.3, 0.4, 0.5?
            2) in addition if src_dx in_degree > 0, connect to dst_cui with same TUI as src_cui if they're within
            same category (diagnostic/procedure/labs/conc)
                example:
                a) if src_cui and dst_cui in conc or dx, connect src_cui and dst_cui with same TUI if src_cui's
                out_degree > 0
                b) if src_cui and dst_cui in PROC, LABS, connect src_cui and dst_cui with same TUI if src_cui's in_degree > 0
            """
            pass
        else:
            raise NotImplementedError(f"{self.mode} method has not been implemented!!")

        return g, m_edges

    @staticmethod
    def _get_subgraphs(g):
        edge_list = list(zip(g.edges()[0].tolist(), g.edges()[1].tolist()))
        nx_g = networkx.from_edgelist(edge_list)
        return list(networkx.connected_components(nx_g))

    def process(self):

        # get semantic info for cui
        sem_file = self._sem_file_path()
        if self.verbose:
            print(f"Reading sem type and group info from {sem_file}...")
            logger.info(f"raw_dir: {self.raw_dir}")
        logger.info(f"sem_file path: {sem_file}")
        sem_info_iter = ProcessedIterExtended(sem_file, header=True, delimiter="\t")


        # load saved embeddings, e.g. KGE .npy file
        if self.verbose:
            print(f"Getting embeding from {self._emb_file_path()}")
        logger.info(f"emb file path: {self._emb_file_path()}")
        cui_ptr_embeddings = np.load(self._emb_file_path())

        for row in sem_info_iter:
            cui, tui, sg = row[1], row[2], row[4]
            self.cui2tui[cui] = tui
            self.cui2sg[cui] = sg

        # get kg rel info for cui
        if "kg_rel" in self.mode:
            self._build_cui2cui_kg_rel_dict()

        # convert dataset.mlb.classes_ to self.glabel_dict mapping idx to actual label classes
        # dataset.mlb.classes_ is an ndarray mapping idx to actual label classes
        self.glabel_dict = {int(label_index[0]): v for label_index, v in np.ndenumerate(self.mlb.classes_)}

        # create graph for each doc in cui doc iter
        for graph_i in range(self.N):
            if (graph_i + 1) % 500 == 0 and self.verbose is True:
                print(f"processing graph {graph_i + 1}...")
            doc_data = self.data[graph_i]
            doc_id, input_ids, input_tokens, glabel, n_nodes = doc_data

            # convert ndarray to Tensor first and append to list
            # avoid converting list of ndarrays to Tensor later, cat along dim=0 instead when batching
            self.labels.append(torch.Tensor(glabel))
            for each_label in self.mlb.inverse_transform(glabel)[0]:
                if each_label not in self._partition_label_dict:
                    self._partition_label_dict[each_label] = len(self._partition_label_dict)

            # build dgl graph store num edges
            g, m_edges = self._build_graph_edges(n_nodes, input_tokens)

            # store node features/embeddings if any
            nlabels = []  # node labels; none for now, can store sem types? sem group? whether cui in icd9 sem type?
            nattrs = []  # node attributes if it has
            nidx_to_cui = dict()  # mapping graph g's node_idx (0-indexed) to cui
            for node_j in range(n_nodes):
                embd_row_idx = input_ids[node_j]
                nattrs.append(cui_ptr_embeddings[embd_row_idx])

                # relabel nodes if it has labels
                # if it doesn't have node labels, then every node's label is its cui semantic tui
                # relabel to indexed 0 to number of semantic group in the dataset - 1
                # this is optional and TODO: this to be refactored as well
                node_cui = input_tokens[node_j]
                node_tui = self.cui2tui[node_cui]
                nidx_to_cui[node_j] = node_cui
                if node_tui not in self.nlabel_dict:
                    mapped = len(self.nlabel_dict)
                    self.nlabel_dict[node_tui] = mapped
                nlabels.append(self.nlabel_dict[node_tui])

            # store node embeddings to the whole graph as torch tensor
            if nattrs != []:
                nattrs = np.stack(nattrs)  # dim[0] == number of nodes/graph, dim[1] embedding dimension e.g. 100
                g.ndata["attr"] = torch.Tensor(nattrs)  # torch.Tensor defaults to dtype float32
                self.nattrs_flag = True

            # store node labels as float32 tensor if any
            # optional
            g.ndata["label"] = torch.Tensor(nlabels)
            if len(self.nlabel_dict) > 1:
                self.nlabels_flag = True

            assert g.num_nodes() == n_nodes

            # update statistics of graphs
            self.n += n_nodes
            self.m += m_edges

            self.graphs.append(g)
            # store node idx to cui mapping for each graph
            # self.gnidx_cuis list order correspond to graph order
            self.gnidx_cuis.append(nidx_to_cui)  # this is retrievable when __getitem__ is called on the class
            # analyze for num subgraphs in each document
            self.cc_dict[graph_i] = self._get_subgraphs(g)
            self.cc += len(self.cc_dict[graph_i])

        # concat labels Tensors in self.labels to be of shape num doc * num label classes
        # from list of torch.Tensors, dtype float32
        self.labels = torch.cat(self.labels, dim=0)
        # if no attr
        if not self.nattrs_flag:
            if self.verbose:
                print("there are no node features in this dataset!")
                # after load, get the #classes and #dim

        self.gclasses = len(self.glabel_dict)
        self.pclasses = len(self._partition_label_dict)
        self.nclasses = len(self.nlabel_dict)
        self.eclasses = len(self.elabel_dict)
        self.dim_nfeats = self.graphs[0].ndata["attr"].shape[-1]  # torch.Size([1,100]) --> 100

        if self.verbose:
            print(f"Done."
                  f"-------- Data Statistics -------- \n"
                  f"#Graphs: {self.N}\n"
                  f"#Graph Classes: {self.gclasses}\n"
                  f"#Partition Label Classes: {self.pclasses}\n"
                  f"#Nodes: {self.n}\n"
                  f"#Node Classes: {self.nclasses}\n"
                  f"#Node Features Dim: {self.dim_nfeats}\n"
                  f"#Edges: {self.m}\n"
                  f"#Edge Classes: {self.eclasses}\n"
                  f"Avg. of #Nodes: {self.n / self.N}\n"
                  f"Avg. of #Edge: {self.m / self.N}\n"
                  f"Avg. of #Subgraphs: {self.cc / self.N}")

    def save(self):
        graph_path = os.path.join(
            self.save_path, f"{self.embedding_type}_{self.name}_{self.hash}.bin"
        )

        info_path = os.path.join(
            self.save_path, f"{self.embedding_type}_{self.name}_{self.hash}.pkl"
        )
        logger.info(f"Saving GNNGraphData and info_dict: {graph_path} and {info_path}...")
        label_dict = {"labels": self.labels}
        info_dict = {
            "gnidx_cui": self.gnidx_cuis,
            "N": self.N,
            "n": self.n,
            "m": self.m,
            "cc": self.cc,
            "self_loop": self.self_loop,
            "gclasses": self.gclasses,
            "pclasses": self.pclasses,
            "nclasses": self.nclasses,
            "eclasses": self.eclasses,
            "dim_nfeats": self.dim_nfeats,
            "glabel_dict": self.glabel_dict,
            "plabel_dict": self._partition_label_dict,
            "nlabel_dict": self.nlabel_dict,
            "elabel_dict": self.elabel_dict,
            "ndegree_dict": self.ndegree_dict,
            "cc_dict": self.cc_dict
        }
        save_graphs(str(graph_path), self.graphs, label_dict)
        save_info(str(info_path), info_dict)

    def load(self):
        graph_path = os.path.join(
            self.save_path, f"{self.embedding_type}_{self.name}_{self.hash}.bin"
        )
        info_path = os.path.join(
            self.save_path, f"{self.embedding_type}_{self.name}_{self.hash}.pkl"
        )
        logger.info(f"Loading from saved graph_path and info_path: {graph_path} and {info_path}")
        graphs, label_dict = load_graphs(str(graph_path))
        info_dict = load_info(str(info_path))

        self.graphs = graphs
        self.labels = label_dict["labels"]
        self.gnidx_cuis = info_dict["gnidx_cui"]

        self.N = info_dict["N"]
        self.n = info_dict["n"]
        self.m = info_dict["m"]
        self.cc = info_dict["cc"]
        self.self_loop = info_dict["self_loop"]
        self.gclasses = info_dict["gclasses"]
        self.pclasses = info_dict["pclasses"]
        self.nclasses = info_dict["nclasses"]
        self.eclasses = info_dict["eclasses"]
        self.dim_nfeats = info_dict["dim_nfeats"]
        self.glabel_dict = info_dict["glabel_dict"]
        self._partition_label_dict = info_dict["plabel_dict"]
        self.nlabel_dict = info_dict["nlabel_dict"]
        self.elabel_dict = info_dict["elabel_dict"]
        self.ndegree_dict = info_dict["ndegree_dict"]
        self.cc_dict = info_dict["cc_dict"]

    def has_cache(self):
        graph_path = os.path.join(
            self.save_path, f"{self.embedding_type}_{self.name}_{self.hash}.bin"
        )
        info_path = os.path.join(
            self.save_path, f"{self.embedding_type}_{self.name}_{self.hash}.pkl"
        )
        if os.path.exists(graph_path) and os.path.exists(info_path):
            logger.info(f"has_cache, will load from saved data at: {graph_path} and {info_path}")
            return True
        else:
            return False

    @property
    def num_classes(self):
        return self.gclasses

    @staticmethod
    def collate_gnn(samples):
        # The input `samples` is a list of tuples
        #  (graph, label, dict mapping node idx to cui).
        graphs, labels, _ = list(zip(*samples))
        batched_graph = dgl.batch(graphs)

        # concat labels Tensors in self.labels to be of shape num doc * num label classes
        # from list of torch.Tensors, dtype float32 --> this step should be done per batch at collate_fn
        # batched_labels = torch.cat(labels, dim=0)

        return batched_graph, torch.stack(labels, dim=0)


if __name__ == '__main__':
    check_gnn_data_reader = False
    check_gnn_dataset = True
    check_gnn_dataloader = False
    if check_gnn_data_reader:
        data_reader = GNNDataReader(data_dir=f"{PROJ_FOLDER / 'data' / 'mimic3'}",
                                    version="full",
                                    input_type="umls",
                                    prune_cui=True,
                                    cui_prune_file="full_cuis_to_discard_snomedcase4.pickle",
                                    vocab_fn="processed_full_umls_pruned.json")
        d_id, x, x_token, y, doc_size = data_reader.get_dataset('dev')[0]
        print(f"id: {d_id}, x: {x}\n, {x_token}, y: {y}, doc_size: {doc_size}")
        train_stats = data_reader.get_dataset_stats("dev")

    if check_gnn_dataset:
        data_reader = GNNDataReader(data_dir=f"{PROJ_FOLDER / 'data' / 'mimic3'}",
                                    version="dummy",
                                    input_type="umls",
                                    prune_cui=True,
                                    cui_prune_file="full_cuis_to_discard_snomedcase4.pickle",
                                    vocab_fn="processed_full_umls_pruned.json")
        gnn_dataset = GNNDataset(dataset=data_reader.get_dataset("train"),
                                 mlb=data_reader.mlb,
                                 name="train",
                                 mode="combined_tui_kg_rel",
                                 force_reload=True,
                                 verbose=True)
        num_samples = len(gnn_dataset)
        g_sample, label_sample, gnidx_sample = gnn_dataset[num_samples - 1]
        print(gnn_dataset.data[num_samples - 1])
        print(g_sample)
        print(label_sample)
        print(gnidx_sample)
        print(isinstance(gnn_dataset, dgl.data.DGLDataset))
        print(isinstance(gnn_dataset, Dataset))

    if check_gnn_dataloader:
        sample_tr_dataloader, emb_size, num_labels = get_dataloader(dataset=gnn_dataset,
                                                                    batch_size=6,
                                                                    shuffle=True,
                                                                    collate_fn=GNNDataset.collate_gnn)
        g_batch, labels_batch = next(iter(sample_tr_dataloader))
        print(g_batch, type(g_batch))
        print(labels_batch)
        print(type(labels_batch))
        print(labels_batch.shape)
    # sem_info_iter = ProcessedIterExtended("../../data/umls/semantic_info.csv", header=True, delimiter="\t")
    # sem_info = list(sem_info_iter)
    # print(sem_info[0])
