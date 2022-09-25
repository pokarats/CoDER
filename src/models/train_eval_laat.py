# -*- coding: utf-8 -*-

import torch
import numpy as np
import random

from torch import optim
from sklearn.metrics import f1_score
from tqdm import tqdm, trange


# get rid of this if using Sacred as it's done by them otherwise pass in their seed param
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
        _run,  # Sacred
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
    n_patience = 6
    evals = list()

    try:
        for epoch_no in trange(epochs, desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
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

            if best_fmicro is None:
                best_fmicro = score

            if score >= best_fmicro:
                best_fmicro = score
                torch.save(model.state_dict(), "./best_{}".format(model_save_fname))

            _run.log_scalar("training/epoch/loss", tr_loss / nb_tr_examples, epoch_no)
            _run.log_scalar("training/epoch/val_loss", eval_data[-1], epoch_no)
            _run.log_scalar("training/epoch/val_score", score, epoch_no)
            evals.append((epoch_no, score, eval_data))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    return evals


def evaluate(dataloader, model, device, no_labels=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    labels_logits, labels_preds, labels = list(), list(), list()

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
                b_inputs, b_labels = batch

            if not no_labels:
                b_labels_logits, b_labels_loss = model(b_inputs, b_labels)
                avg_loss += b_labels_loss.item()
            else:
                b_labels_logits = model(b_inputs)

            b_labels_preds = (torch.sigmoid(b_labels_logits).detach().cpu().numpy() >= 0.5).astype(int)
            b_labels_logits = detach(b_labels_logits, float)

            if not no_labels:
                b_labels = detach(b_labels, int)

            labels_preds = append(labels_preds, b_labels_preds)
            labels_logits = append(labels_logits, b_labels_logits)

            if not no_labels:
                labels = append(labels, b_labels)

    labels_preds = labels_preds[0]
    labels_logits = labels_logits[0]

    if not no_labels:
        labels = labels[0]
        avg_loss /= len(dataloader)
    ids = [items[0] for items in dataloader.dataset.data]

    mlb = dataloader.dataset.mlb
    if not no_labels:
        final_labels, final_preds = normalize_labels(labels, labels_preds, mlb)
        mlb_labels = mlb.transform(final_labels)
        mlb_preds = mlb.transform(final_preds)
        score = f1_score(y_true=mlb_labels, y_pred=mlb_preds, average='micro')
        print("\nEvaluation - loss: {:.6f}  f1: {:.4f}%\n".format(avg_loss, score * 100))
    else:
        score = 0.
        return score, (labels_logits, mlb.inverse_transform(labels_preds), None, ids, avg_loss)

    return score, (labels_logits, final_preds, final_labels, ids, avg_loss)


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
    """
    for i in range(len(labels)):
        l, lp = labels[i], labels_preds[i]
        l = np.nonzero(l)[0]
        lp = np.nonzero(lp)[0]
        final_labels.append([f'{mlb.classes_[a]}' for a in l])
        final_preds.append([f'{mlb.classes_[a]}' for a in lp])
    """

    return final_labels, final_preds


def generate_preds_file(preds, preds_ids, preds_file):
    docid2preds = dict()
    for idx in range(len(preds)):
        docid2preds[preds_ids[idx]] = preds[idx]

    with open(preds_file, "w") as wf:
        for doc_id, preds in docid2preds.items():
            doc_id = doc_id.strip()
            if preds == ['NONE'] or preds == []:
                line = str(doc_id) + "\n"
            else:
                line = str(doc_id) + "\t" + "|".join(preds) + "\n"
            wf.write(line)

    return preds
