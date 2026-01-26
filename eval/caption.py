import re
import json
import argparse
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from bert_score import BERTScorer, score
from pycocoevalcap.cider.cider import Cider


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--answer_file', type=str)
    args = parser.parse_args()

    with open(args.answer_file, 'r') as f:
        answer_data_list = json.load(f)

    truth_dict = {}
    pred_dict = {}
    truth_list = []
    pred_list = []
    for answer_data in tqdm(answer_data_list):
        truth = answer_data['conversations'][1]['value']
        pred = answer_data['conversations'][1]['generative_value']
        truth_dict[answer_data['id']] = [truth]
        pred_dict[answer_data['id']] = [pred]
        truth_list.append(truth)
        pred_list.append(pred)
    
    cider = Cider()
    cider_score, _ = cider.compute_score(truth_dict, pred_dict)
    print("CIDEr: ", cider_score)

    # P, R, F1 = score(pred_list, truth_list, lang='en', verbose=True, device='cuda:0')
    # bert_score = F1.sum() / len(F1)
    # print("BERTScore: ", bert_score.item())
    