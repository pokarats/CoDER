#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION: WIP -- Access point for GNN experiments.

@copyright: Copyright 2018 Deutsches Forschungszentrum fuer Kuenstliche
            Intelligenz GmbH or its licensors, as applicable.

@author: Noon Pokaratsiri Goldstein, adapted from Saadullah Amin's LAAT implementation
(https://github.com/suamin/P4Q_Guttmann_SCT_Coding/blob/main/laat.py)



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

from models.gnn import GCNGraphClassification
from models.train_eval_laat import train, evaluate, generate_preds_file
from utils.prepare_gnn_data import GNNDataReader, GNNDataset
from src.utils.corpus_readers import get_data, MimicCuiSelectedTextIter
from utils.config import PROJ_FOLDER, MODEL_FOLDER, DEV_API_KEY
from utils.eval import all_metrics
from neptune.new.integrations.sacred import NeptuneObserver
from sacred.observers import FileStorageObserver
from sacred import Experiment
import neptune.new as neptune
from neptune.exceptions import NeptuneException

SAVED_FOLDER = PROJ_FOLDER / f"scratch/.log/{date.today():%y_%m_%d}/{Path(__file__).stem}"

# Step 1: Initialize Neptune and create new Neptune run
neptune_run = neptune.init(
    project="GraphStructuresMimicMLC/MimicICD9",
    api_token=DEV_API_KEY,
    tags=f"gnn snomed ct"
)

# Step 2: Add NeptuneObserver() to your sacred experiment's observers
ex = Experiment()
ex.observers.append(FileStorageObserver(SAVED_FOLDER))
ex.observers.append(NeptuneObserver(run=neptune_run))


@ex.capture
def load_model(_log,
               embedding_path,
               cui_embedding_path,
               batch_size,
               dr_params,
               gnn_params):
    _log.info(f"dr_params: {dr_params}")
    _log.info(f"Pre_trained embedding weights from {embedding_path}")
    # embed_matrix = np.load(f"{embedding_path}")
    # w2v_weights_from_np = torch.Tensor(embed_matrix)

    # for combined text and cui model
    if cui_embedding_path is not None:
        raise NotImplementedError(f"Combined inputs not implemented for GNN Model")
    else:
        dr, train_data_loader, dev_data_loader, test_data_loader = get_data(batch_size,
                                                                            GNNDataset,
                                                                            GNNDataset.collate_gnn,
                                                                            GNNDataReader,
                                                                            **dr_params)
    train_data_loader, emb_size, num_labs = train_data_loader
    dev_data_loader, _, _ = dev_data_loader
    test_data_loader, _, _ = test_data_loader

    # TODO: Vocab size in terms of GNN Dataset??
    _log.info(f"Embedding Dim: {emb_size}\n"
              f"Num Labels: {num_labs}\n")

    _log.info(f"Dataset Stats\n"
              f"Train Partition ({len(train_data_loader.dataset)} samples):\n{dr.get_dataset_stats('train')}\n"
              f"Dev Partition ({len(dev_data_loader.dataset)} samples):\n{dr.get_dataset_stats('dev')}\n"
              f"Test Partition ({len(test_data_loader.dataset)} samples):\n{dr.get_dataset_stats('test')}\n")

    if cui_embedding_path is not None:
        raise NotImplementedError("pending implementation!")

    else:
        gnn_params = {"de": emb_size,
                      "L": num_labs,
                      **gnn_params}
        _log.info(f"gnn_params: {gnn_params}")
        model = GCNGraphClassification(**gnn_params)

    return model, train_data_loader, dev_data_loader, test_data_loader


# Sacred logging parameters
@ex.config
def gnn_cfg():
    """Default params for training on 50/full MIMIC-III CUI GNN datasets"""
    # Directory and data organization
    # data directory and file organization
    data_dir = PROJ_FOLDER / "data"
    version = "50"
    input_type = "umls"
    embedding_type = "umls"  # options are: umls, snomedbase, snomedcase4, somednoex
    mimic_dir = Path(data_dir) / "mimic3" / f"{version}"
    doc_iterator = None
    cui_embedding_path = None

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
    if input_type == "text":
        embedding_path = model_dir / "processed_full_text_pruned.npy"
    elif input_type == "umls":
        embedding_path = model_dir / f"processed_full_{embedding_type}_pruned.npy"

    else:
        # combined embedding types
        embedding_path = Path(mimic_dir).parent / "model" / "processed_full_text_pruned.npy"
        cui_embedding_path = model_dir / f"processed_full_{embedding_type}_pruned.npy"

    # LAAT and GNN model params, n, de, L, pre_trained_weights defined in load_model captured function
    gnn_params = dict(u=256,
                      da=256,
                      dropout=0.3,
                      num_layers=2,
                      readout='mean')

    # DataReader class params, first arg is batch_size
    if doc_iterator is not None:
        dr_params = dict(data_dir=f"{Path(data_dir) / 'mimic3'}",
                         version=version,
                         input_type=input_type,
                         prune_cui=False,
                         cui_prune_file=None,
                         vocab_fn=f"processed_full_{input_type}_pruned.json",
                         max_seq_length=4000,
                         doc_iterator=MimicCuiSelectedTextIter,
                         umls_iterator=None)
    else:
        dr_params = dict(data_dir=f"{Path(data_dir) / 'mimic3'}",
                         version=version,
                         input_type=input_type,
                         prune_cui=True,
                         cui_prune_file=f"{version}_cuis_to_discard_{embedding_type}.pickle",
                         vocab_fn=f"processed_full_{input_type}_pruned.json",
                         max_seq_length=None,
                         doc_iterator=None,
                         umls_iterator=None,
                         embedding_type=embedding_type,
                         mode="gcn_base",
                         verbose=True)


@ex.named_config
def dummy_cfg():
    """Dummy params for testing on dummy MIMIC-III text datasets"""
    # Directory and data organization
    # data directory and file organization
    data_dir = PROJ_FOLDER / "data"
    version = "dummy"
    input_type = "umls"
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


@ex.main
def run_gnn(embedding_path,
            cui_embedding_path,
            batch_size,
            dr_params,
            gnn_params,
            device,
            epochs,
            lr,
            early_stop,
            grad_clip,
            eval_only,
            _log,
            _run):
    model, train_data_loader, dev_data_loader, test_data_loader = load_model(_log,
                                                                             embedding_path=embedding_path,
                                                                             cui_embedding_path=cui_embedding_path,
                                                                             batch_size=batch_size,
                                                                             dr_params=dr_params,
                                                                             gnn_params=gnn_params)
    model = model.to(device)
    _log.info(f"loaded model info:\n{model}")
    version = dr_params["version"]
    model_mode = dr_params.get('mode', 'non_gnn')
    model_save_fname = f"filtered_{version}_{dr_params['input_type']}_{Path(embedding_path).stem}_{model_mode}"

    if not eval_only:
        _log.info(f"{'=' * 10}{model_mode.upper()} TRAINING STARTED{'=' * 10}")
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

    _log.info(f"{'=' * 10}{model_mode.upper()} EVALUATION STARTED{'=' * 10}")
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
    predicted_fp = f"{MODEL_FOLDER / f'{version}_{model_mode.upper()}_test_preds_{exp_vers}.txt'}"
    generate_preds_file(test_eval_data["final_predicted"],
                        test_eval_data["doc_ids"],
                        preds_file=predicted_fp)
    try:
        _run.add_artifact(predicted_fp, name="predicted_labels_file.txt")
    except NeptuneException as ne:
        _log.exception(f"{ne} artifact was saved in {SAVED_FOLDER} but not on Neptune", stack_info=True)

    return dict(final_training_loss=final_tr_loss,
                final_training_f1=final_tr_f1,
                eval_loss=eval_loss,
                eval_f1=eval_f1,
                **test_eval_metrics)


# Step 3: Run you experiment and explore metadata in the Neptune app
if __name__ == '__main__':
    ex.run_commandline()
