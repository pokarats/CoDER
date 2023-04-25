"""
Adapted from examples from https://docs.dgl.ai/en/rying_test/guide/minibatch.html#guide-minibatch
"""
import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from tqdm import tqdm


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


# GCN model for graph classification
# from GDL example
class GCNGraphClassification(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCNGraphClassification, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats)
        self.conv2 = dgl.nn.GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = nn.relu(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")


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
