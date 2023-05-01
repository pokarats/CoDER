import torch
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import save_graphs, save_info, load_graphs, load_info
import os
import itertools
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from src.utils.corpus_readers import MimicDocIter, MimicCuiDocIter, MimicCuiSelectedTextIter, ProcessedIterExtended
from src.utils.prepare_laat_data import DataReader

os.environ['DGLBACKEND'] = 'pytorch'

"""
class DataReader:

    def __init__(self,
                 data_dir="../../data/mimic3",
                 version="full",
                 input_type="text",
                 prune_cui=False,
                 cui_prune_file=None,
                 vocab_fn="processed_full_text_pruned.json",
                 max_seq_length=4000,
                 doc_iterator=None,
                 umls_iterator=None,
                 second_txt_vocab_fn=None):
"""


class GNNDataReader(DataReader):
    """
    Read datafile(s) from both text .csv files and pre-processed lined_data dataset files for CUI inputs and prepare
    data for GNNDataset class.

    1) Each sample will be returned as list of unique CUIs
    2) MLB for each sample labels
    3) Calculate stats for each partition
    """
    def __init__(self,
                 data_dir="../../data/mimic3",
                 version="50",
                 input_type="umls",
                 prune_cui=True,
                 cui_prune_file="50_cuis_to_discard_snomedcase4.pickle",
                 vocab_fn="processed_full_text_pruned.json",
                 max_seq_length=None,
                 doc_iterator=None,
                 umls_iterator=None,
                 second_txt_vocab_fn=None):
        super().__init__(data_dir=data_dir,
                         version=version,
                         input_type=input_type,
                         prune_cui=prune_cui,
                         cui_prune_file=cui_prune_file,
                         vocab_fn=vocab_fn,
                         max_seq_length=max_seq_length,
                         doc_iterator=doc_iterator,
                         umls_iterator=umls_iterator,
                         second_txt_vocab_fn=second_txt_vocab_fn)
        self.doc2len = dict()

    def _fit_transform(self, split):
        # only need CUIs for now
        # return doc ID, iput CUIs, only unique CUIs, doc len == num of unique CUIs that have been flattened
        if self.input_type == "text":
            raise NotImplementedError(f"Invalud input_type: only CUI input is supported!")
        elif "umls" in self.input_type:
            umls_doc_iter = self.umls_doc_iterator(self.umls_doc_split_path[split],
                                                   threshold=0.7,
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

                yield doc_id, input_ids, list(unique_tokens_only), self.mlb.transform([doc_labels]), len(unique_tokens_only)
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


class GNNDataset(DGLDataset):

    def __init__(self,
                 dataset,
                 mlb,
                 name="train",
                 emb_name="snomedcase4",
                 self_loop=True,
                 raw_dir="../../data",
                 save_dir="../../data/gnn_data",  # save_path = os.path.join(save_dir, self.name)
                 force_reload=False,
                 verbose=False,
                 transform=None,
                 ):

        self._name = name  # MIMIC-III-CUI train/dev/test partition
        self.ds_name = "mimic3_cui"
        self.emb_name = emb_name

        self.data = dataset  # DataReader.get_dataset('<split: train/dev/test>')
        # self.id2label = {k: v for k, v in enumerate(self.mlb.classes_)}
        self.mlb = mlb  # has already been fit with all label classes

        self.cui2tui = dict()  # mapping cui to tui from semantic_info.csv <--\t separated, col 2
        self.cui2sg = dict()  # mapping cui to semantic group from semantic_info.csv <-- \t separated col 4

        self.self_loop = self_loop
        self.graphs = []
        self.labels = []

        # label dict mapping idx to labels?
        self.glabel_dict = {}  # mapping of mlb.classes_ idx to original labels, classes_ global across train/test/dev
        self._partition_label_dict = {}  # mapping idx to actual label classe in this partition only
        self.nlabel_dict = {}
        self.elabel_dict = {}
        self.ndegree_dict = {}

        # global num
        self.N = len(self.data)  # total graphs number
        self.n = 0  # total nodes number
        self.m = 0  # total edges number

        # global num of classes
        self.gclasses = 0  # number of classes in the whole dataset
        self.pclasses = 0  # number of classes in this partition
        self.nclasses = 0
        self.eclasses = 0
        self.dim_nfeats = 0

        # flags
        self.nattrs_flag = False
        self.nlabels_flag = False

        super(GNNDataset, self).__init__(
            name=name,
            hash_key=(name, emb_name, self_loop),
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    @property
    def raw_path(self):
        return os.path.join(self.raw_dir, "gnn_data")

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, index):
        """Get the idx-th sample.
                Parameters
                ---------
                idx : int
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
        return g, self.labels[index]
        # doc_id, input_ids, labels_bin = self.data[index]
        # return input_ids, labels_bin

    def _sem_file_path(self):
        return os.path.join(
            self.raw_dir,
            "umls",
            "semantic_info.csv"
        )

    def _emb_file_path(self):
        return os.path.join(
            self.raw_dir,
            "linked_data",
            "model",
            f"processed_full_{self.emb_name}_pruned.npy"
        )

    def process(self):

        # get semantic info for cui
        sem_file = self._sem_file_path()
        sem_info_iter = ProcessedIterExtended(sem_file, header=True, delimiter="\t")

        # load saved embeddings, e.g. KGE .npy file
        cui_ptr_embeddings = np.load(self._emb_file_path())

        if self.verbose:
            print(f"Reading sem type and group info from {sem_file}...")
        for row in sem_info_iter:
            cui = row[1]
            tui = row[2]
            sg = row[4]
            self.cui2tui[cui] = tui
            self.cui2sg[cui] = sg

        # convert dataset.mlb.classes_ to self.glabel_dict mapping idx to actual label classes
        # dataset.mlb.classes_ is an ndarray mapping idx to actual label classes
        self.glabel_dict = {int(label_index[0]): v for label_index, v in np.ndenumerate(self.mlb.classes_)}

        # create graph for each doc in cui doc iter
        for graph_i in range(self.N):
            if (graph_i + 1) % 10 == 0 and self.verbose is True:
                print(f"processing graph {graph_i + 1}...")
            doc_data = self.data[graph_i]
            doc_id, input_ids, input_tokens, glabel, n_nodes = doc_data

            # convert ndarray to Tensor first and append to list
            # avoid converting list of ndarrays to Tensor later, cat along dim=0 instead when batching
            self.labels.append(torch.Tensor(glabel))
            for each_label in self.mlb.inverse_transform(glabel)[0]:
                if each_label not in self._partition_label_dict:
                    self._partition_label_dict[each_label] = len(self._partition_label_dict)

            # create dgl graph for each doc
            g = dgl.graph(([], []))
            g.add_nodes(n_nodes)

            nlabels = []  # node labels; none for now, can store sem types? sem group? whether cui in icd9 sem type?
            nattrs = []  # node attributes if it has
            m_edges = 0

            # add edges
            # TODO: refactor this to accomodate future improvements baseline vs different ways of connecting edges
            if self.self_loop:
                # product includes self element
                groupby_iterable = itertools.product(range(n_nodes), repeat=2)
            else:
                # permute doesn't include self element
                groupby_iterable = itertools.permutations(range(n_nodes), r=2)
            for src_dst_pair in groupby_iterable:
                src_idx, dst_idx = src_dst_pair
                src_cui, dst_cui = input_tokens[src_idx], input_tokens[dst_idx]
                if self.cui2tui[src_cui] == self.cui2tui[dst_cui]:
                    m_edges += 1
                    g.add_edges(src_idx, dst_idx)

            # store node features/embeddings if any
            for node_j in range(n_nodes):
                embd_row_idx = input_ids[node_j]
                nattrs.append(cui_ptr_embeddings[embd_row_idx])

                # relabel nodes if it has labels
                # if it doesn't have node labels, then every node's label is its cui semantic group
                # relabel to indexed 0 to number of semantic group in the dataset - 1
                # this is optional and TODO: this to be refactored as well
                node_cui = input_tokens[node_j]
                node_sg = self.cui2sg[node_cui]
                if node_sg not in self.nlabel_dict:
                    mapped = len(self.nlabel_dict)
                    self.nlabel_dict[node_sg] = mapped
                nlabels.append(self.nlabel_dict[node_sg])

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

        # concat labels Tensors in self.labels to be of shape num doc * num label classes
        # from list of torch.Tensors, dtype float32 --> this step should be done per batch at collate_fn
        # self.labels = torch.cat(self.labels, dim=0)
        # if no attr
        if not self.nattrs_flag:
            if self.verbose:
                print("there are no node features in this dataset!")
                # after load, get the #classes and #dim

        self.gclasses = len(self.glabel_dict)
        self.pclasses = len(self._partition_label_dict)
        self.nclasses = len(self.nlabel_dict)
        self.eclasses = len(self.elabel_dict)
        self.dim_nfeats = len(self.graphs[0].ndata["attr"][1])

        if self.verbose:
            print(f"Done."
                  f"-------- Data Statistics -------- \n"
                  f"#Graphs: {self.N}\n"
                  f"#Graph Classes: {self.gclasses}\n"
                  f"#Partition Label Classes: {self.pclasses}\n"
                  f"#Nodes: {self.n}\n"
                  f"#Node Classes: {self.nclasses}\n"
                  f"#Node Features Dim: {self.dim_nfeats}"
                  f"#Edges: {self.m}\n"
                  f"#Edge Classes: {self.eclasses}\n"
                  f"Avg. of #Nodes: {self.n / self.N}\n"
                  f"Avg. of #Edge: {self.m / self.N}\n")
            # TODO: add avg #subgraphs (i.e. connected components logic) dgl <-->networkx
            # g = networkx.from_edgelist(same_sem_types)
            # cc = networkx.connected_components(g)
            #

    def save(self):
        graph_path = os.path.join(
            self.save_path, f"{self.emb_name}_{self.name}_{self.hash}.bin"
        )

        info_path = os.path.join(
            self.save_path, f"{self.emb_name}_{self.name}_{self.hash}.pkl"
        )
        label_dict = {"labels": self.labels}
        info_dict = {
            "N": self.N,
            "n": self.n,
            "m": self.m,
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
        }
        save_graphs(str(graph_path), self.graphs, label_dict)
        save_info(str(info_path), info_dict)

    def load(self):
        graph_path = os.path.join(
            self.save_path, f"{self.emb_name}_{self.name}_{self.hash}.bin"
        )
        info_path = os.path.join(
            self.save_path, f"{self.emb_name}_{self.name}_{self.hash}.pkl"
        )
        graphs, label_dict = load_graphs(str(graph_path))
        info_dict = load_info(str(info_path))

        self.graphs = graphs
        self.labels = label_dict["labels"]

        self.N = info_dict["N"]
        self.n = info_dict["n"]
        self.m = info_dict["m"]
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

    def has_cache(self):
        graph_path = os.path.join(
            self.save_path, f"{self.emb_name}_{self.name}_{self.hash}.bin"
        )
        info_path = os.path.join(
            self.save_path, f"{self.emb_name}_{self.name}_{self.hash}.pkl"
        )
        if os.path.exists(graph_path) and os.path.exists(info_path):
            return True
        else:
            return False

    @property
    def num_classes(self):
        return self.gclasses

    @staticmethod
    def collate_gnn(samples):
        # The input `samples` is a list of pairs
        #  (graph, label).
        graphs, labels = list(zip(*samples))
        batched_graph = dgl.batch(graphs)

        # concat labels Tensors in self.labels to be of shape num doc * num label classes
        # from list of torch.Tensors, dtype float32 --> this step should be done per batch at collate_fn
        batched_labels = torch.cat(labels, dim=0)

        return batched_graph, batched_labels


def get_dataloader(dataset, batch_size, shuffle, collate_fn=GNNDataset.collate_gnn, num_workers=8):
    data_loader = GraphDataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers
    )
    return data_loader


def get_data(batch_size=8, dataset_class=GNNDataset, collate_fn=GNNDataset.collate_gnn, **kwargs):
    dr = DataReader(**kwargs)
    train_data_loader = get_dataloader(dataset_class(dr.get_dataset('train'), dr.mlb), batch_size, True, collate_fn)
    dev_data_loader = get_dataloader(dataset_class(dr.get_dataset('dev'), dr.mlb), batch_size, False, collate_fn)
    test_data_loader = get_dataloader(dataset_class(dr.get_dataset('test'), dr.mlb), batch_size, False, collate_fn)
    return dr, train_data_loader, dev_data_loader, test_data_loader


if __name__ == '__main__':
    check_gnn_data_reader = False
    check_gnn_dataset = True
    if check_gnn_data_reader:
        data_reader = GNNDataReader(data_dir="../../data/mimic3",
                                    version="full",
                                    input_type="umls",
                                    prune_cui=True,
                                    cui_prune_file="full_cuis_to_discard_snomedcase4.pickle",
                                    vocab_fn="processed_full_umls_pruned.json")
        d_id, x, x_token, y, doc_size = data_reader.get_dataset('train')[0]
        print(f"id: {d_id}, x: {x}\n, {x_token}, y: {y}, doc_size: {doc_size}")
        train_stats = data_reader.get_dataset_stats("train")

    if check_gnn_dataset:
        data_reader = GNNDataReader(data_dir="../../data/mimic3",
                                    version="full",
                                    input_type="umls",
                                    prune_cui=True,
                                    cui_prune_file="full_cuis_to_discard_snomedcase4.pickle",
                                    vocab_fn="processed_full_umls_pruned.json")
        gnn_dataset = GNNDataset(dataset=data_reader.get_dataset("train"),
                                 mlb=data_reader.mlb,
                                 name="train",
                                 verbose=True)
        g_sample, label_sample = gnn_dataset[0]
        print(g_sample)
        print(label_sample)


    # sem_info_iter = ProcessedIterExtended("../../data/umls/semantic_info.csv", header=True, delimiter="\t")
    # sem_info = list(sem_info_iter)
    # print(sem_info[0])
