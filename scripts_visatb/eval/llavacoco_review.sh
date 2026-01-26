#!/bin/bash

CKPT=""

python eval_gpt_review_visual.py \
    --question qa90_questions.jsonl \
    --context caps_boxes_coco2014_val_80.jsonl \
    --rule rule.json \
    --answer-list \
        qa90_gpt4_answer.jsonl \
        answers/$CKPT.jsonl \
    --output reviews/$CKPT.jsonl

python summarize_gpt_review.py -f reviews/$CKPT.jsonl
