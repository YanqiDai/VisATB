#!/bin/bash

CKPT=""

python -m llava.eval.model_vqa \
    --model-path ./checkpoints/$CKPT \
    --question-file ./playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder /home/yanqi_dai/data/3MT_Instruct/images/coco/val2014 \
    --answers-file ./playground/data/eval/llava-coco/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1