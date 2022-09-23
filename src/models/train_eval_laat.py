# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import random
import argparse

from torch import optim
from sklearn.metrics import f1_score
from tqdm import tqdm, trange
from laat import LAAT
from src.utils.prepare_laat_data import get_data


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)

def train(
        train_dataloader,
        dev_dataloader,
        model,
        epochs,
        lr,
        device,
        decay_rate=1.0,
        grad_clip=None,
        model_save_fname="model.pt"
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if decay_rate > 0.:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay_rate)
    else:
        scheduler = None
    steps = 0
    best_fmicro = None
    last_fmicro = None
    n_patience = 10
    evals = list()

    try:
        for epoch_no in trange(epochs, desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                inputs, labels, presents = batch

                labels_logits, labels_loss, presents_logits, presents_loss = model(inputs, labels, presents)
                loss = labels_loss + (1.0 * presents_loss)
                loss.backward()

                tr_loss += labels_loss.item() + presents_loss.item()
                nb_tr_examples += inputs.size(0)
                nb_tr_steps += 1

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()
                steps += 1

            if scheduler is not None:
                scheduler.step()

            score, eval_data = evaluate(dev_dataloader, model, device)

            if best_fmicro is None:
                best_fmicro = score

            if score >= best_fmicro:
                best_fmicro = score
                torch.save(model.state_dict(), "./best_{}".format(model_save_fname))

            evals.append((epoch_no, score, eval_data))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    return evals


def evaluate(dataloader, model, device, no_labels=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    labels_logits, labels_preds, labels = list(), list(), list()
    presents_logits, presents_preds, presents = list(), list(), list()
    ids = list()
    avg_loss = 0.

    def append(all_tensors, batch_tensor):
        if len(all_tensors) == 0:
            all_tensors.append(batch_tensor)
        else:
            all_tensors[0] = np.append(all_tensors[0], batch_tensor, axis=0)
        return all_tensors

    def detach(tensor, dtype=None):
        if dtype:
            return tensor.detach().cpu().numpy().astype(dtype)
        else:
            return tensor.detach().cpu().numpy()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(device) for t in batch)
            if no_labels:
                b_inputs = batch
            else:
                b_inputs, b_labels, b_presents = batch

            if not no_labels:
                b_labels_logits, b_labels_loss, b_presents_logits, b_presents_loss = model(b_inputs, b_labels,
                                                                                           b_presents)
                avg_loss += b_labels_loss.item() + b_presents_loss.item()
            else:
                b_labels_logits, b_presents_logits = model(b_inputs)

            b_labels_preds = (torch.sigmoid(b_labels_logits).detach().cpu().numpy() >= 0.5).astype(int)
            b_labels_logits = detach(b_labels_logits, float)

            b_presents_preds = torch.max(torch.softmax(b_presents_logits, dim=1), 1)[1].detach().cpu().numpy().astype(
                int)
            b_presents_logits = detach(b_presents_logits, float)

            if not no_labels:
                b_labels = detach(b_labels, int)
                b_presents = detach(b_presents, int)

            labels_preds = append(labels_preds, b_labels_preds)
            presents_preds = append(presents_preds, b_presents_preds)
            labels_logits = append(labels_logits, b_labels_logits)
            presents_logits = append(presents_logits, b_presents_logits)
            if not no_labels:
                labels = append(labels, b_labels)
                presents = append(presents, b_presents)

    labels_preds = labels_preds[0]
    labels_logits = labels_logits[0]
    presents_preds = presents_preds[0]
    presents_logits = presents_logits[0]
    if not no_labels:
        labels = labels[0]
        presents = presents[0]
        avg_loss /= len(dataloader)
    ids = [items[0] for items in dataloader.dataset.data]

    if not no_labels:
        final_labels, final_preds = normalize_labels(
            labels, presents, labels_preds, presents_preds,
            dataloader.dataset.id2label,
        )
        mlb_labels = dataloader.dataset.mlb.transform(final_labels)
        mlb_preds = dataloader.dataset.mlb.transform(final_preds)
        score = f1_score(y_true=mlb_labels, y_pred=mlb_preds, average='micro')
        print("\nEvaluation - loss: {:.6f}  f1: {:.4f}%\n".format(avg_loss, score * 100))
    else:
        score = 0.

    return score, ((labels_logits, presents_logits), final_preds, final_labels, ids, avg_loss)


def normalize_labels(labels, presents, labels_preds, presents_preds, id2label):
    final_labels = list()
    final_preds = list()
    transform = {0: "0", 1: "1"}
    for i in range(len(labels)):
        l, p, lp, pp = labels[i], presents[i], labels_preds[i], presents_preds[i]
        l = np.nonzero(l)[0]
        p = [p[j] for j in l]
        lp = np.nonzero(lp)[0]
        pp = [pp[j] for j in lp]
        final_labels.append([f'{id2label[a]}_{transform[b]}' for a, b in list(zip(l, p))])
        final_preds.append([f'{id2label[a]}_{transform[b]}' for a, b in list(zip(lp, pp)) if b != 2])
    return final_labels, final_preds


def generate_preds_file(id2label, preds, preds_ids, preds_file):
    docid2preds = dict()
    for idx in range(len(preds)):
        docid2preds[preds_ids[idx]] = preds[idx]

    with open(preds_file, "w") as wf:
        for doc_id, preds in docid2preds.items():
            doc_id = doc_id.strip()
            if preds == ['NONE']:
                line = str(doc_id) + "\n"
            else:
                line = str(doc_id) + "\t" + "|".join(preds) + "\n"
            wf.write(line)

    return preds