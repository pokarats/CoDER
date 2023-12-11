"""
Adapted from examples from https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.explain.GNNExplainer.html
"""
import torch
import dgl
from dgl.nn.pytorch import GNNExplainer
from utils.config import PROJ_FOLDER
from utils.corpus_readers import get_data
from utils.prepare_gnn_data import GNNDataReader, GNNDataset
from models.gnn import GCNGraphClassification


DATA_DIR = f"{PROJ_FOLDER / 'data' / 'mimic3'}"


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load dummy dataset and dataloaders
    dummy_dr, dummy_tr, dummy_dev, dummy_test = get_data(batch_size=2,
                                                         dataset_class=GNNDataset,
                                                         collate_fn=GNNDataset.collate_gnn,
                                                         reader=GNNDataReader,
                                                         data_dir=DATA_DIR,
                                                         version="dummy",
                                                         mode="base_kg_rel",
                                                         input_type="umls",
                                                         prune_cui=True,
                                                         cui_prune_file="full_cuis_to_discard_snomedcase4.pickle",
                                                         vocab_fn="processed_full_umls_pruned.json")
    tr_loader, dim_nfeats, num_label_classes = dummy_tr
    dev_loader, _, _ = dummy_dev
    test_loader, _, _ = dummy_test

    sample_g, sample_labels, gnidx_to_cui = dev_loader.dataset[0]
    print(sample_g)
    print(sample_labels)

    # Create the model with given dimensions
    model = GCNGraphClassification(de=dim_nfeats,
                                   u=256,
                                   da=256,
                                   L=num_label_classes,
                                   dropout=0.3,
                                   num_layers=1,
                                   readout='sum')
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):
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
                print(type(inputs[0]))
                print(f"num in batched graph: {inputs[0].batch_size}")

            print(f"Epoch_batch: {epoch}_{batch_i} -- Loss: {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f"GCNClassification Model working!!")

    # Test GNN Explainer
    explainer = GNNExplainer(model, num_hops=1)
    sample_features = sample_g.ndata['attr']
    feat_mask, edge_mask = explainer.explain_graph(sample_g, sample_features)
    print(feat_mask)
    """
    feat_mask is a tensor of dim==node feature dimension,
    """
    print(edge_mask)
    """
    edge_mask is a tensor of dim==number of edges,
    np.where[edge_mask > condition e.g. 0.7][0] --> array idx of edges with scores > condition
    the i-th idx is an EID, which with g.find_edges(EID or [EIDs]) will give src, dst node ids
    """

