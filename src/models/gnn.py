import torch
import torch.nn as nn
import dgl


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
