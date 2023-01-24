"""
GNN training adapted from link_prediction on GraphSAGE example in
https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/link_pred.py
"""
import torch
import torch.nn as nn
from torch import optim
from dgl.dataloading import DataLoader, NeighborSampler, negative_sampler, as_edge_prediction_sampler
from tqdm import tqdm, trange
import logging
from src.utils.config import MODEL_FOLDER


logger = logging.getLogger(__name__)

if not MODEL_FOLDER.exists():
    MODEL_FOLDER.mkdir(parents=True, exist_ok=False)


def train(args,
          device,
          g,
          reverse_eids,
          seed_edges,
          model,
          _run,
          lr=0.0005,
          epochs=10,
          decay_rate=0.9):
    # create sampler & dataloader
    sampler = NeighborSampler([15, 10, 5], prefetch_node_feats=['feat'])
    sampler = as_edge_prediction_sampler(sampler,
                                         exclude='reverse_id',
                                         reverse_eids=reverse_eids,
                                         negative_sampler=negative_sampler.Uniform(1))
    use_uva = (args.mode == 'mixed')
    dataloader = DataLoader(g,
                            seed_edges,
                            sampler,
                            device=device,
                            batch_size=512,
                            shuffle=True,
                            drop_last=False,
                            num_workers=0,
                            use_uva=use_uva)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0)
    # per LAAT, decay default=0 and optim should only update params that requires_grad

    if decay_rate > 0.:
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay_rate)
        # reduce 10% if stagnant for 5 epochs per LAAT paper
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode="max",
                                                         patience=5,
                                                         factor=decay_rate,
                                                         min_lr=0.0001,
                                                         verbose=True)
    else:
        scheduler = None

    for epoch_no in trange(epochs, desc="Epoch"):
        model.train()
        total_loss = 0
        for step, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(dataloader):
            x = blocks[0].srcdata['feat']
            pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)

            score = torch.cat([pos_score, neg_score])

            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            labels = torch.cat([pos_label, neg_label])

            loss = nn.BCEWithLogitsLoss(score, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (step+1) == 1000:
                break
        logger.info(f"Epoch {epoch_no} | Loss {total_loss / (step+1):.4f}")
        logger.info("Epoch {:05d} | Loss {:.4f}".format(epoch_no, total_loss / (step+1)))
        # Sacred/Neptune logging
        # _run.log_scalar("training/epoch/loss", total_loss / nb_tr_examples, epoch_no)
