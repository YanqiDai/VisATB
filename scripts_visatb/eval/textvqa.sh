#!/bin/bash

IMAGE_FOLDER=""
CKPT="./checkpoints"

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/$CKPT \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $IMAGE_FOLDER \
    --answers-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl