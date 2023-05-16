"""
Adapted from examples from https://docs.dgl.ai/en/rying_test/guide/minibatch.html#guide-minibatch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from src.utils.config import PROJ_FOLDER
from src.utils.corpus_readers import get_data
from src.utils.prepare_gnn_data import GNNDataReader, GNNDataset
from tqdm import tqdm

DATA_DIR = f"{PROJ_FOLDER / 'data' / 'mimic3'}"


class GCNGraphClassification(nn.Module):
    """
    Baseline 2-layer GCN Model with 1-fc layer for Graph Classification adapted from DGL examples and tutorials
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/train.py
    https://github.com/liketheflower/dgl_examples/tree/cc42d8e00314bd4efb1fb4e8f1167ffb01ffff14/graph_classification

    Added BCELogitsLoss func for when labels are provided, otherwise forward func only returns prediction logits

    TODO: 1) add option to specity aggregator besides mean 2) batch norm? 3) integrate Label-Attention from LAAT?
    """
    def __init__(self, de, u, da, L, dropout=0.3):
        super(GCNGraphClassification, self).__init__()
        self.conv1 = dgl.nn.GraphConv(de, u)
        self.conv2 = dgl.nn.GraphConv(u, da)
        self.dropout = nn.Dropout(dropout)
        self.labels_output = nn.Linear(da, L, bias=True)
        self.labels_loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, g, y=None):
        # in_feat is the node embedding, each node represents CUI, so embedding size == 100 as in LAAT
        # g is a graph_batch from GraphDataloader
        g, in_feat = g, g.ndata["attr"]
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h  # b x num nodes x h_feats
        # print(f"h.size, {h.size()}")
        # graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, "h")  # b x h_feats
        # print(f"hg.size, {hg.size()}")
        labels_output = self.labels_output(hg)  # b x num_classes
        # print(f"output size: {labels_output.size()}")
        output = (labels_output,)
        if y is not None:
            # print("y size:", y.size())
            loss = self.labels_loss_fct(labels_output, y)  # .sum(-1).mean()
            output += (loss,)
        return output


"""
===========================================================================
Codes below are still under development and are not currently being used!!!
===========================================================================
"""


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'))
            return edge_subgraph.edata['score']


class HeteroGraphScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']


class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv2 = dgl.nn.GraphConv(hidden_features, out_features)

    def forward(self, blocks, x):
        x = nn.relu(self.conv1(blocks[0], x))
        x = nn.relu(self.conv2(blocks[1], x))
        return x


class StochasticTwoLayerRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
        super().__init__()
        self.conv1 = dgl.nn.HeteroGraphConv({rel: dgl.nn.GraphConv(in_feat,
                                                                   hidden_feat,
                                                                   norm='right')
                                             for rel in rel_names})
        self.conv2 = dgl.nn.HeteroGraphConv({rel: dgl.nn.GraphConv(hidden_feat,
                                                                   out_feat,
                                                                   norm='right')
                                             for rel in rel_names})

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = self.conv2(blocks[1], x)
        return x


class GCNModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_classes=None, edge_type=None):
        super().__init__()
        if edge_type is not None:
            # heterogenous RGCN
            self.gcn = StochasticTwoLayerRGCN(in_features, hidden_features, out_features, edge_type)
            self.predictor = HeteroGraphScorePredictor()
        else:
            self.gcn = StochasticTwoLayerGCN(in_features, hidden_features, out_features)
            self.predictor = ScorePredictor()
        self.labels_loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, positive_graph, negative_graph, blocks, x):
        x = self.gcn(blocks, x)
        pos_score = self.predictor(positive_graph, x)
        neg_score = self.predictor(negative_graph, x)

        return pos_score, neg_score


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        h = x
        h = nn.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


# GIN model for graph classification from DGL Graph classification example in
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/train.py
class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = nn.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer


"""
Snippets below are adapted from https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/link_pred.py
"""


class GraphSAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size=1):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dgl.nn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dgl.nn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dgl.nn.SAGEConv(hid_size, hid_size, 'mean'))
        self.hid_size = hid_size
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, out_size))

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for layer_num, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if layer_num != len(self.layers) - 1:
                h = nn.relu(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)
        for lalyer_num, layer in enumerate(self.layers):
            y = torch.empty(g.num_nodes(), self.hid_size, device=buffer_device,
                            pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc='Inference'):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if lalyer_num != len(self.layers) - 1:
                    h = nn.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load dummy dataset and dataloaders
    dummy_dr, dummy_tr, dummy_dev, dummy_test = get_data(batch_size=2,
                                                         dataset_class=GNNDataset,
                                                         collate_fn=GNNDataset.collate_gnn,
                                                         reader=GNNDataReader,
                                                         data_dir=DATA_DIR,
                                                         version="dummy",
                                                         input_type="umls",
                                                         prune_cui=True,
                                                         cui_prune_file="full_cuis_to_discard_snomedcase4.pickle",
                                                         vocab_fn="processed_full_umls_pruned.json")
    tr_loader, dim_nfeats, num_label_classes = dummy_tr
    dev_loader, _, _ = dummy_dev
    test_loader, _, _ = dummy_test

    # Create the model with given dimensions
    model = GCNGraphClassification(de=dim_nfeats, u=256, da=256, L=num_label_classes, dropout=0.3)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):
        for batch_i, batch in enumerate(tr_loader):
            batch = tuple(tensor.to(device) for tensor in batch)
            *inputs, labels = batch
            pred_logits, loss = model(*inputs, labels)
            print(f"Epoch_batch: {epoch}_{batch_i} -- Pred:\n{pred_logits}")
            print(f"num in batch: {pred_logits.size(0)}")
            try:
                print(f"num in batch_graph: {inputs[0].size(0)}")
            except AttributeError:
                print("To get DGLHeteroGraph batch size:\n")
                print(f"num in batched graph: {inputs[0].batch_size}")

            print(f"Epoch_batch: {epoch}_{batch_i} -- Loss: {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f"GCNClassification Model working!!")
