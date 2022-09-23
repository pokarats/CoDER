# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import random
import argparse

from torch import optim
from sklearn.metrics import f1_score
from tqdm import tqdm, trange
from src.models.laat import LAAT
from src.models.train_eval_laat import train, evaluate, generate_preds_file
from src.utils.prepare_laat_data import get_data


def main(device, batch_size=16, max_seq_length=384, epochs=50, lr=0.0005, eval_only=False, use_focus_concept=True):
    embed_matrix = np.load(os.path.join('models/word2vec', 'word2vec.guttmann.100d_mat.npy'))
    dr, train_data_loader, dev_data_loader, test_data_loader = get_data(batch_size=batch_size, max_seq_length=max_seq_length)

    model = LAAT(len(dr.featurizer.vocab), embed_matrix.shape[1], len(dr.id2label))
    model = model.to(device)
    print(model)
    model.word_embed.weight.data.copy_(torch.from_numpy(embed_matrix))
    model_save_fname = "{}_{}.pt".format('guttmann', 'LAAT')

    if not eval_only:
        train(
            train_data_loader, dev_data_loader, model, epochs, lr,
            device=device, grad_clip=None, model_save_fname=model_save_fname
        )

    model.load_state_dict(torch.load('./best_guttmann_LAAT.pt'))

    _, (_, preds, _, ids, _) = evaluate(test_data_loader, model, device)

    return model, model_save_fname, preds, ids, dr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use_focus_concept", action="store_true",
        help="Whether to only consider focused concepts."
    )

    args = parser.parse_args()

    import pprint
    pprint.pprint(vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, model_save_fname, dev_preds, preds_ids, dr = main(
        device=device,
        use_focus_concept=args.use_focus_concept
    )
    torch.save(model.state_dict(), './' + model_save_fname)
    # generate predictions file for evaluation script
    generate_preds_file(
        dr.id2label, dev_preds, preds_ids,
        preds_file="./preds_test.txt"
    )

    eval_cmd = """$python evaluation.py \
            --ids_file='{}' \
            --anns_file='{}' \
            --dev_file='{}' \
            --out_file='{}'"""
    if args.use_focus_concept:
        eval_cmd += " --use_focus_concept"
    eval_cmd = eval_cmd.format(
        "data/clef_format/test/ids_test.txt",
        "data/clef_format/test/anns_test.txt" ,
        "preds_test.txt",
        "eval_output.txt"
    )
    eval_results = os.popen(eval_cmd).read()
    print("eval results with challenge script:")
    print(eval_results)