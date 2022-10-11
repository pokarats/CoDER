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

from models.laat import LAAT
from models.train_eval_laat import train, evaluate, generate_preds_file
from utils.prepare_laat_data import get_data
from utils.config import PROJ_FOLDER, MODEL_FOLDER, DEV_API_KEY
from utils.eval import all_metrics
from neptune.new.integrations.sacred import NeptuneObserver
from sacred.observers import FileStorageObserver
from sacred import Experiment
import neptune.new as neptune


SAVED_FOLDER = PROJ_FOLDER / f"scratch/.log/{date.today():%y_%m_%d}/{Path(__file__).stem}"

# Step 1: Initialize Neptune and create new Neptune run
neptune_run = neptune.init(
    project="pokarats/LAAT",
    api_token=DEV_API_KEY,
    tags=f"slurm top50 text"
)

# Step 2: Add NeptuneObserver() to your sacred experiment's observers
ex = Experiment()
ex.observers.append(FileStorageObserver(SAVED_FOLDER))
ex.observers.append(NeptuneObserver(run=neptune_run))


@ex.capture
def load_model(_log,
               embedding_path,
               batch_size,
               dr_params,
               laat_params):

    _log.info(f"Loading pre_trained embedding weights from {embedding_path}")
    embed_matrix = np.load(f"{embedding_path}")
    w2v_weights_from_np = torch.Tensor(embed_matrix)

    dr, train_data_loader, dev_data_loader, test_data_loader = get_data(batch_size,
                                                                        **dr_params)
    _log.info(f"Vocab size: {len(dr.featurizer.vocab)}\n"
              f"Embedding Dim: {embed_matrix.shape[1]}\n"
              f"Num Labels: {len(dr.mlb.classes_)}\n")

    _log.info(f"Dataset Stats\n"
              f"Train Partition ({len(train_data_loader.dataset)} samples):\n{dr.get_dataset_stats('train')}\n"
              f"Dev Partition ({len(dev_data_loader.dataset)} samples):\n{dr.get_dataset_stats('dev')}\n"
              f"Test Partition ({len(test_data_loader.dataset)} samples):\n{dr.get_dataset_stats('test')}\n")

    laat_params = {"n": len(dr.featurizer.vocab),
                   "de": embed_matrix.shape[1],
                   "L": len(dr.mlb.classes_),
                   "pre_trained_weights": w2v_weights_from_np,
                   **laat_params}

    model = LAAT(**laat_params)

    return model, train_data_loader, dev_data_loader, test_data_loader


# Sacred logging parameters
@ex.config
def text_cfg():
    """Default params for training on 50/full MIMIC-III text datasets"""
    # Directory and data organization
    # data directory and file organization
    data_dir = PROJ_FOLDER / "data"
    version = "50"
    input_type = "text"
    mimic_dir = Path(data_dir) / "mimic3" / f"{version}"

    if input_type == "text":
        model_dir = Path(mimic_dir).parent / "model"
    else:
        # input_type == "umls"
        model_dir = Path(data_dir) / "linked_data" / "model"

    # Pytorch hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 50
    lr = 0.001
    early_stop = False
    grad_clip = None
    eval_only = False

    # load model params are separate into sections below
    batch_size = 8
    embedding_path = model_dir / f"processed_full_{input_type}_pruned.npy"

    # LAAT model params, n, de, L, pre_trained_weights defined in load_model captured function
    laat_params = dict(u=256,
                       da=256,
                       dropout=0.3,
                       pad_idx=0,
                       trainable=False)  # word embedding weights static

    # DataReader class params, first arg is batch_size
    dr_params = dict(data_dir=f"{Path(data_dir) / 'mimic3'}",
                     version=version,
                     input_type=input_type,
                     prune_cui=False,
                     cui_prune_file=None,
                     vocab_fn=f"processed_full_{input_type}_pruned.json",
                     max_seq_length=4000,
                     doc_iterator=None,
                     umls_iterator=None)


@ex.named_config
def dummy_cfg():
    """Dummy params for testing on dummy MIMIC-III text datasets"""
    # Directory and data organization
    # data directory and file organization
    data_dir = PROJ_FOLDER / "data"
    version = "dummy"
    input_type = "text"
    mimic_dir = Path(data_dir) / "mimic3" / f"{version}"

    if input_type == "text":
        model_dir = Path(mimic_dir).parent / "model"
    else:
        # input_type == "umls"
        model_dir = Path(data_dir) / "linked_data" / "model"

    # Pytorch hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 3
    lr = 0.001
    early_stop = False
    grad_clip = None
    eval_only = False

    # load model params are separate into sections below
    batch_size = 2

    # LAAT model params, n, de, L, pre_trained_weights defined in load_model captured function
    # unchanged

    # DataReader class params, first arg is batch_size
    # unchanged from defaults


def run_eval_pred():
    pass


@ex.main
def run_laat(embedding_path,
             batch_size,
             dr_params,
             laat_params,
             device,
             epochs,
             lr,
             early_stop,
             grad_clip,
             eval_only,
             _log,
             _run):

    model, train_data_loader, dev_data_loader, test_data_loader = load_model(embedding_path=embedding_path,
                                                                             batch_size=batch_size,
                                                                             dr_params=dr_params,
                                                                             laat_params=laat_params)
    model = model.to(device)
    print(model)
    version = dr_params["version"]
    model_save_fname = f"{version}_{Path(embedding_path).stem}_LAAT"

    if not eval_only:
        _log.info(f"{'=' * 10}LAAT TRAINING STARTED{'=' * 10}")
        tr_ep_num, tr_f1_scores, tr_eval_data = zip(*train(train_data_loader,
                                                    dev_data_loader,
                                                    model,
                                                    epochs,
                                                    lr,
                                                    device=device,
                                                    _run=_run,
                                                    early_stop=early_stop,
                                                    grad_clip=grad_clip,
                                                    model_save_fname=model_save_fname))

    _log.info(f"{'=' * 10}LAAT EVALUATION STARTED{'=' * 10}")
    saved_model_path = MODEL_FOLDER / f'best_{model_save_fname}.pt'
    _log.info(f"Loading best model state from {saved_model_path}")
    model.load_state_dict(torch.load(f"{saved_model_path}"))

    eval_f1, test_eval_data = evaluate(test_data_loader, model, device)
    eval_loss = test_eval_data["avg_loss"]

    # eval metrics
    test_raw = test_eval_data["logits"]
    test_y = test_eval_data["true_labels"]
    test_y_hat = test_eval_data["predicted"]
    test_eval_metrics = all_metrics(test_y_hat, test_y, k=[5, 8, 15], yhat_raw=test_raw, calc_auc=True)

    _run.log_scalar("testing/eval_loss", eval_loss)
    _run.log_scalar("testing/eval_f1_micro", eval_f1)
    _run.log_scalar("testing/eval_f1_macro", test_eval_metrics["f1_macro"])
    _run.log_scalar("testing/eval_P@5", test_eval_metrics["prec_at_5"])
    _run.log_scalar("testing/eval_AUC_micro", test_eval_metrics["auc_micro"])
    _run.log_scalar("testing/eval_AUC_macro", test_eval_metrics["auc_macro"])

    if not eval_only:
        final_tr_loss = tr_eval_data[-1]["avg_loss"]
        final_tr_f1 = tr_f1_scores[-1]
    else:
        final_tr_loss = None
        final_tr_f1 = None

    # generate predictions file for evaluation script
    exp_vers = Path(embedding_path).stem
    predicted_fp = f"{MODEL_FOLDER / f'{version}_LAAT_test_preds_{exp_vers}.txt'}"
    generate_preds_file(test_eval_data["final_predicted"],
                        test_eval_data["doc_ids"],
                        preds_file=predicted_fp)
    _run.add_artifact(predicted_fp, name="predicted_labels_file")

    return dict(final_training_loss=final_tr_loss,
                final_training_f1=final_tr_f1,
                eval_loss=eval_loss,
                eval_f1=eval_f1,
                **test_eval_metrics)


# Step 3: Run you experiment and explore metadata in the Neptune app
if __name__ == '__main__':
    ex.run_commandline()
