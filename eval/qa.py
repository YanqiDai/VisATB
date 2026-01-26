import json
import argparse
from tqdm import tqdm

def match_pred_truth(pred, truth):
    pred = pred.strip().lower()
    truth = truth.strip().lower()
    if len(pred) > 0 and pred[-1] == '.':
        pred = pred[:-1]
    if len(truth) > 0 and truth[-1] == '.':
        truth = truth[:-1]
    if pred == truth:
        return True
    else:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--answer_file', type=str)
    args = parser.parse_args()

    with open(args.answer_file, 'r') as f:
        answer_data_list = json.load(f)
    
    score = 0

    for answer_data in tqdm(answer_data_list):
        truth = answer_data['conversations'][1]['value']
        pred = answer_data['conversations'][1]['generative_value']
        if match_pred_truth(pred, truth):
            score += 1

    print(score, "/", len(answer_data_list), "=", score / len(answer_data_list))