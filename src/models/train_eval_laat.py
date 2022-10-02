# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
import logging

from torch import optim
from sklearn.metrics import f1_score
from tqdm import tqdm, trange

import os
import platform
import sys
from pathlib import Path
from src.utils.config import MODEL_FOLDER


# get rid of this if using Sacred as it's done by them otherwise pass in their seed param
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# set_seed(42)  # uncomment this if not running with Sacred

logger = logging.getLogger(__name__)

if not MODEL_FOLDER.exists():
    MODEL_FOLDER.mkdir(parents=True, exist_ok=False)


def train(
        train_dataloader,
        dev_dataloader,
        model,
        epochs,
        lr,
        device,
        _run,  # Sacred metrics api
        early_stop=False,
        decay_rate=1.0,
        grad_clip=None,
        model_save_fname="LAAT_model"
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if decay_rate > 0.:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay_rate)
    else:
        scheduler = None
    steps = 0
    best_fmicro = None
    last_fmicro = None
    n_patience = 6
    evals = list()

    try:
        for epoch_no in trange(epochs, desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Train Batch Iteration")):
                batch = tuple(tensor.to(device) for tensor in batch)
                inputs, labels = batch

                labels_logits, labels_loss = model(inputs, labels)
                loss = labels_loss
                if step % 50 == 0:
                    _run.log_scalar("training/batch/loss", loss, step)
                tr_loss += labels_loss.item()

                loss.backward()
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

            # first epoch
            if best_fmicro is None:
                best_fmicro = score
            if last_fmicro is None:
                last_fmicro = score

            if score >= best_fmicro:
                best_fmicro = score
                logger.info(f"saving best model so far from epoch: {epoch_no}, micro f1: {score}\n")
                torch.save(model.state_dict(), f"{MODEL_FOLDER / f'best_{model_save_fname}.pt'}")

            if early_stop:
                if score < last_fmicro:
                    n_patience -= 1
                    if n_patience == 0:
                        logger.info(f"No. tolerance reached for worse score than last!\n"
                                    f"Early stopping triggered in {epoch_no} epochs\n")
                        evals.append((epoch_no, score, eval_data))
                        break

            # Sacred/Neptune logging
            _run.log_scalar("training/epoch/loss", tr_loss / nb_tr_examples, epoch_no)
            _run.log_scalar("training/epoch/val_loss", eval_data[-1], epoch_no)
            _run.log_scalar("training/epoch/val_score", score, epoch_no)

            last_fmicro = score
            evals.append((epoch_no, score, eval_data))

    except KeyboardInterrupt:
        print('*' * 20)
        print('Exiting from training early')

    return evals


def evaluate(dataloader, model, device, no_labels=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    labels_logits, labels_preds, labels = list(), list(), list()

    dataset_doc_ids = [items[0] for items in dataloader.dataset.data]
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
        for batch in tqdm(dataloader, desc="Eval Batch Iteration"):
            batch = tuple(tensor.to(device) for tensor in batch)
            if no_labels:
                b_inputs = batch
            else:
                b_inputs, b_labels = batch  # b_inputs torch.int64, b_labels torch.float32

            if not no_labels:
                b_labels_logits, b_labels_loss = model(b_inputs, b_labels)
                avg_loss += b_labels_loss.item()
            else:
                b_labels_logits = model(b_inputs)

            # predicted labels dtype int
            b_labels_preds = (torch.sigmoid(b_labels_logits).detach().cpu().numpy() >= 0.5).astype(int)
            b_labels_logits = detach(b_labels_logits, float)

            if not no_labels:
                b_labels = detach(b_labels, int)  # detached labels dtype int

            labels_preds = append(labels_preds, b_labels_preds)
            labels_logits = append(labels_logits, b_labels_logits)

            if not no_labels:
                labels = append(labels, b_labels)

    labels_preds = labels_preds[0]
    labels_logits = labels_logits[0]

    if not no_labels:
        labels = labels[0]
        avg_loss /= len(dataloader)

    # get MultilabelBinarized that's been fit to inverse transform labels
    mlb = dataloader.dataset.mlb
    if not no_labels:
        final_labels, final_preds = normalize_labels(labels, labels_preds, mlb)
        score = f1_score(y_true=labels, y_pred=labels_preds, average='micro')
        logger.info(f"\nEvaluation - loss: {avg_loss:.6f}  f1: {score * 100:.4f}\n")
    else:
        score = 0.
        eval_data = {"logits": labels_logits,
                     "predicted": mlb.inverse_transform(labels_preds),
                     "true_labels": None,
                     "doc_ids": dataset_doc_ids,
                     "avg_loss": avg_loss}

        # score = micro f1
        return score, eval_data

    eval_data = {"logits": labels_logits,
                 "predicted": final_preds,
                 "true_labels": final_labels,
                 "doc_ids": dataset_doc_ids,
                 "avg_loss": avg_loss}

    return score, eval_data


def normalize_labels(labels, labels_preds, mlb):
    """
    Get pre-binarized labels back for labels and predicted labels. mlb MUST already be pre-fitted!!
    :param labels: labels for dataset partition, iterable of iterable len(labels.shape) > 1
    :type labels:
    :param labels_preds: predicted labels for dataset partition, iterable of iterable len(labels.shape) > 1
    :type labels_preds:
    :param mlb: sklearn MultilabelBinarizer that has been fit to ALL possible labels for this dataset
    :type mlb:
    :return: ([(label1, label2, ...),(label1, label2, ...), ...], [(label1, label2, ...),(label1, label2, ...), ...])

    """
    final_labels = mlb.inverse_transform(labels)
    final_preds = mlb.inverse_transform(labels_preds)

    return final_labels, final_preds


def generate_preds_file(preds, pred_doc_ids, preds_file):
    """

    :param preds: non-binarized version of predicted labels for the dataset partition
    :type preds: iterable of labels
    :param pred_doc_ids: doc_ids for the dataset partition (should be what's used in dataloader.dataset
    :type pred_doc_ids: iterable of ids
    :param preds_file: path to file for saving preds
    :type preds_file: str or Path
    :return: predicted labels in non-binarized format
    :rtype: iterable of iterables
    """
    docid2preds = dict()
    for idx in range(len(preds)):
        docid2preds[pred_doc_ids[idx]] = preds[idx]

    with open(preds_file, "w") as wf:
        for doc_id, preds in docid2preds.items():
            doc_id = doc_id.strip()
            if not preds or preds == ['NONE']:
                # empty labels e.g. empty [] or empty ()
                print(f"{doc_id}", end="\n", file=wf)
                # line = str(doc_id) + "\n"
            else:
                print(f"{doc_id}\t{'|'.join(preds)}", end="\n", file=wf)
                # line = str(doc_id) + "\t" + "|".join(preds) + "\n"
        logger.info(f"Predictions saved to {preds_file}\n")

    return preds
