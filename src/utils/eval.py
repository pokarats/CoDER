# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# Adopted from:
#
# https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py
#
# and
#
# https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network/blob/master/utils.py
# ----------------------------------------------------------------------------------------------------------------

import numpy as np
import os
import logging

from pathlib import Path
from sklearn.metrics import roc_curve, auc


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__file__)


def union_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)


def intersect_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)


def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)


def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1


def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)


def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)


def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)


def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)


def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1


def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), \
           micro_recall(yhatmic, ymic), micro_f1(yhatmic, ymic)


def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    # get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        # only if there are true positives for this label
        if y[:,i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)
    
    # macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)
    
    # micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic)
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])
    
    return roc_auc


def recall_at_k(yhat_raw, y, k):
    # num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]
    
    # get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        denom = y[i,:].sum()
        vals.append(num_true_in_top_k / float(denom))
    
    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.
    
    return np.mean(vals)


def precision_at_k(yhat_raw, y, k):
    # num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]
    
    # get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i,tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))
    
    return np.mean(vals)


def all_metrics(yhat, y, k=8, yhat_raw=None, calc_auc=True, 
                hierarchical=False, hierarchy_path=None,
                hier_dist=10000, hier_err=5):
    """
        Inputs:
            yhat: binary predictions matrix
            y: binary ground truth matrix
            k: for @k metrics
            yhat_raw: prediction scores matrix (floats)
        Outputs:
            dict holding relevant metrics
    """
    names = ["acc", "prec", "rec", "f1"]
    
    # macro
    macro = all_macro(yhat, y)
    
    # micro
    ymic = y.ravel()
    yhatmic = yhat.ravel()
    micro = all_micro(yhatmic, ymic)
    
    metrics = {names[i] + "_macro": macro[i] for i in range(len(macro))}
    metrics.update({names[i] + "_micro": micro[i] for i in range(len(micro))})
    
    if hierarchical:
        metrics.update(
            hierarchical_metrics(y, yhat, hierarchy_path, hier_dist, hier_err)
        )
    
    # AUC and @k
    if yhat_raw is not None and calc_auc:
        # allow k to be passed as int or list
        if type(k) != list:
            k = [k]
        for k_i in k:
            rec_at_k = recall_at_k(yhat_raw, y, k_i)
            metrics['rec_at_%d' % k_i] = rec_at_k
            prec_at_k = precision_at_k(yhat_raw, y, k_i)
            metrics['prec_at_%d' % k_i] = prec_at_k
            metrics['f1_at_%d' % k_i] = 2*(prec_at_k*rec_at_k)/(prec_at_k+rec_at_k)
        
        roc_auc = auc_metrics(yhat_raw, y, ymic)
        metrics.update(roc_auc)
    
    return metrics


def convert_y_to_str(y):
    return '\n'.join([' '.join(list(map(str, y[i].nonzero()[0].tolist()))) for i in range(y.shape[0])])


def hierarchical_metrics(y, yhat, hierarchy_path, hier_dist=10000, hier_err=5):
    fpath = os.path.split(os.path.abspath(__file__))[0]
    hemkit_dir = Path(fpath) / "HEMKit"
    
    if not os.path.exists(hemkit_dir):
        curdir = os.path.curdir
        os.system(f'wget http://bioasq.org/resources/software/HEMKit.zip')
        os.system(f'mv HEMKit.zip {fpath}')
        os.system(f'unzip {fpath}/HEMKit.zip -d {fpath}')
        os.chdir(hemkit_dir / "software")
        os.system('make')
        os.chdir(curdir)
        os.remove(f'{fpath}/HEMKit.zip')
    
    # HEMKit expects files as input so we create two temporary files
    # for gold and predicted labels
    with open('y.txt', 'w') as wf:
        wf.write(convert_y_to_str(y))
    
    with open('yhat.txt', 'w') as wf:
        wf.write(convert_y_to_str(yhat))
    
    hemkit_bin = Path(hemkit_dir) / "bin" / "HEMKit"
    
    eval_cmd = f'{hemkit_bin} {hierarchy_path} y.txt yhat.txt {hier_dist} {hier_err}'
    eval_results = os.popen(eval_cmd).read()
    """
    Is like this e.g.
    =================
        Hierarchical Precision = 0.952381
        Hierarchical Recall = 0.703704
        Hierarchical F1  = 0.788462
        SDL with all ancestors = 3
        LCA F = 0.777778
        LCA P = 0.916667
        LCA R = 0.7
        MGIA = 0.711111
    """
    hier_metrics = {}
    for idx, line in enumerate(eval_results.strip().split('\n')):
        metric_str = line.split(' = ')
        if idx == 0:
            hier_metrics['hierarchical_precision'] = float(metric_str[1])
        if idx == 1:
            hier_metrics['hierarchical_recall'] = float(metric_str[1])
        if idx == 2:
            hier_metrics['hierarchical_f1'] = float(metric_str[1])
        if idx == 3:
            hier_metrics['sdl'] = float(metric_str[1])
        if idx == 4:
            hier_metrics['lca_f1'] = float(metric_str[1])
        if idx == 5:
            hier_metrics['lca_precision'] = float(metric_str[1])
        if idx == 6:
            hier_metrics['lca_recall'] = float(metric_str[1])
        if idx == 7:
            hier_metrics['mgia'] = float(metric_str[1])
    os.remove('y.txt')
    os.remove('yhat.txt')
    return hier_metrics


def log_metrics(metrics):
    print()
    if "auc_macro" in metrics.keys():
        logger.info("[MACRO] accuracy, precision, recall, f-measure, AUC")
        logger.info(
            "%.4f, %.4f, %.4f, %.4f, %.4f" % (
                metrics["acc_macro"], metrics["prec_macro"], 
                metrics["rec_macro"], metrics["f1_macro"], 
                metrics["auc_macro"]
            )
        )
    else:
        logger.info("[MACRO] accuracy, precision, recall, f-measure")
        logger.info(
            "%.4f, %.4f, %.4f, %.4f" % (
                metrics["acc_macro"], metrics["prec_macro"], 
                metrics["rec_macro"], metrics["f1_macro"]
            )
        )
    
    if "auc_micro" in metrics.keys():
        logger.info("[MICRO] accuracy, precision, recall, f-measure, AUC")
        logger.info(
            "%.4f, %.4f, %.4f, %.4f, %.4f" % (
                metrics["acc_micro"], metrics["prec_micro"], 
                metrics["rec_micro"], metrics["f1_micro"], 
                metrics["auc_micro"]
            )
        )
    else:
        logger.info("[MICRO] accuracy, precision, recall, f-measure")
        logger.info(
            "%.4f, %.4f, %.4f, %.4f" % (
                metrics["acc_micro"], metrics["prec_micro"], 
                metrics["rec_micro"], metrics["f1_micro"]
            )
        )
    
    if "hierarchical_f1" in metrics.keys():
        logger.info("[HIERARCHICAL] precision, recall, f-measure")
        logger.info(
            "%.4f, %.4f, %.4f" % (
                metrics["hierarchical_precision"], metrics["hierarchical_recall"], 
                metrics["hierarchical_f1"]
            )
        )
        logger.info("[LCA] precision, recall, f-measure")
        logger.info(
            "%.4f, %.4f, %.4f" % (
                metrics["lca_precision"], metrics["lca_recall"], 
                metrics["lca_f1"]
            )
        )
        logger.info("SDL: %.4f" % metrics["sdl"])
        logger.info("MGIA: %.4f" % metrics["mgia"])
    
    for metric, val in metrics.items():
        if metric.find("rec_at") != -1:
            logger.info("%s: %.4f" % (metric, val))
    print()


def test():
    y = np.array([
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1]
    ], dtype=np.float32)
    yhat = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1]
    ], dtype=np.float32)
    yhat_raw = np.array([
        [0.21, 0.85, 0.11, 0.03, 0.01],
        [0.33, 0.25, 0.98, 0.20, 0.75],
        [0.13, 0.23, 0.89, 0.27, 0.78],
    ], dtype=np.float32)
    dummy_hier = np.array([
        [10, 20], 
        [20,  1],
        [20,  2],
        [10, 30],
        [30,  5],
        [30,  4],
        [90, 30],
        [70,  3],
        [10, 70],
        [30,  3]
    ])
    with open("hier.txt", "w") as wf:
        wf.write("\n".join([" ".join(list(map(str, line))) for line in dummy_hier]))
    metrics = all_metrics(
        yhat, y, k=[1, 3, 5], yhat_raw=yhat_raw, calc_auc=True,
        hierarchical=True, hierarchy_path='hier.txt'
    )
    os.remove('hier.txt')
    log_metrics(metrics)


if __name__=="__main__":
    test()
