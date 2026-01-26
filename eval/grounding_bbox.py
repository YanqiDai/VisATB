import re
import json
import argparse
from tqdm import tqdm
import torch
from torchvision.ops.boxes import box_area

def get_grounding_box(grounding):
    grounding = grounding.replace('[', '').replace(']', '').split(',')
    grounding = [float(x.strip()) for x in grounding]
    grounding = torch.tensor(grounding)
    return grounding

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union

def match_pred_truth(pred, truth):
    pattern = r'\[(.*?)\]'
    truth_match = re.search(pattern, truth)
    turth_grounding = truth_match.group(1) if truth_match else "[0.0, 0.0, 0.0, 0.0]"  # 返回匹配的子串
    pred_match = re.search(pattern, pred)
    pred_grounding = pred_match.group(1) if pred_match else "[0.0, 0.0, 0.0, 0.0]"  # 返回匹配的子串
    turth_grounding = get_grounding_box(turth_grounding)
    pred_grounding = get_grounding_box(pred_grounding)
    iou, union = box_iou(turth_grounding.unsqueeze(0), pred_grounding.unsqueeze(0))
    if iou > 0.5:
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