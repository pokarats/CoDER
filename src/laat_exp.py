#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: WIP

@copyright: Copyright 2018 Deutsches Forschungszentrum fuer Kuenstliche
            Intelligenz GmbH or its licensors, as applicable.

@author: Noon Pokaratsiri Goldstein, adapted from Saadullah Amin's LAAT implementation
(https://github.com/suamin/P4Q_Guttmann_SCT_Coding/blob/main/laat.py)

LAAT Model as proposed by Vu et al. 2020 (https://www.ijcai.org/proceedings/2020/461)

"""

import torch
import numpy as np
import os
import platform
import sys
from pathlib import Path
from datetime import date


if platform.system() != 'Darwin':
    sys.path.append(os.getcwd())  # only needed for slurm

from src.models.laat import LAAT
from src.models.train_eval_laat import train, evaluate, generate_preds_file
from src.utils.prepare_laat_data import get_data

from neptune.new.integrations.sacred import NeptuneObserver
from sacred.observers import FileStorageObserver
from sacred import Experiment
import neptune.new as neptune

PROJ_FOLDER = Path(__file__).resolve().parent.parent
SAVED_FOLDER = PROJ_FOLDER / f"scratch/.log/{date.today():%y_%m_%d}/{Path(__file__).stem}"
MODEL_FOLDER = PROJ_FOLDER / "res" / f"{date.today():%y_%m_%d}"

# Step 1: Initialize Neptune and create new Neptune run
# TODO: need .env.dev file and dot_env lib to set and load api token env var
neptune_run = neptune.init(
    project="pokarats/LAAT",
    api_token=os.environ.get("NEPTUNE_API_TOKEN"),
    tags=f"{date.today():%y_%m_%d}"
)

# Step 2: Add NeptuneObserver() to your sacred experiment's observers
ex = Experiment()
ex.observers.append(FileStorageObserver(SAVED_FOLDER))
ex.observers.append(NeptuneObserver(run=neptune_run))


# Log hyperparameters
@ex.config
def cfg():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    max_seq_length = 384
    epochs = 50
    lr = 0.001
    eval_only = False

    embedding_npy_path = "path"
    u = 256
    da = 256
    dropout = 0.3

    pad_idx = 0
    trainable = True


@ex.post_run_hook
def run_eval_pred():
    pass


@ex.main
def run_laat(device,
             batch_size,
             max_seq_length,
             epochs,
             lr,
             eval_only,
             embedding_npy_path,
             u,
             da,
             dropout,
             pad_idx,
             trainable,
             _log,
             _run):

    embed_matrix = np.load(f"{embedding_npy_path}")
    w2v_weights_from_np = torch.Tensor(embed_matrix)
    dr, train_data_loader, dev_data_loader, test_data_loader = get_data(batch_size=batch_size,
                                                                        max_seq_length=max_seq_length)

    model = LAAT(n=len(dr.featurizer.vocab),
                 de=embed_matrix.shape[1],
                 L=len(dr.id2label),
                 u=u,
                 da=da,
                 dropout=dropout,
                 pad_idx=pad_idx,
                 pre_trained_weights=w2v_weights_from_np,
                 trainable=trainable)

    model = model.to(device)
    print(model)
    model_save_fname = f"{Path(embedding_npy_path).stem}_LAAT"

    if not eval_only:
        _log.info(f"{'=' * 10}LAAT TRAINING STARTED{'=' * 10}")
        tr_ep_num, tr_f1_scores, tr_eval_data = zip(*train(train_data_loader,
                                                    dev_data_loader,
                                                    model,
                                                    epochs,
                                                    lr,
                                                    device=device,
                                                    _run=_run,
                                                    grad_clip=None,
                                                    model_save_fname=model_save_fname))

    _log.info(f"{'=' * 10}LAAT EVALUATION STARTED{'=' * 10}")
    saved_model_path = MODEL_FOLDER / f'best_{model_save_fname}.pt'
    _log.info(f"Loading best model state from {saved_model_path}")
    model.load_state_dict(torch.load(f"{saved_model_path}"))

    eval_f1, test_eval_data = evaluate(test_data_loader, model, device)
    eval_loss = test_eval_data["avg_loss"]

    if not eval_only:
        final_tr_loss = tr_eval_data["avg_loss"]
        final_tr_f1 = tr_f1_scores[-1]
    else:
        final_tr_loss = None
        final_tr_f1 = None

    # generate predictions file for evaluation script
    exp_vers = Path(embedding_npy_path).stem
    predicted_fp = f"{MODEL_FOLDER / f'LAAT_test_preds_{exp_vers}.txt'}"
    generate_preds_file(test_eval_data["predicted"],
                        test_eval_data["doc_ids"],
                        preds_file=predicted_fp)
    _run.add_artifact(predicted_fp, name="predicted_labels_file")

    return dict(final_training_loss= final_tr_loss,
                final_training_f1=final_tr_f1,
                eval_loss=eval_loss,
                eval_f1=eval_f1)


# Step 3: Run you experiment and explore metadata in the Neptune app
if __name__ == '__main__':
    ex.run_commandline()
