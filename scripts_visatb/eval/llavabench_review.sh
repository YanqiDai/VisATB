#!/bin/bash

CKPT=""

python eval_gpt_review_bench.py \
    --question llava-bench-in-the-wild/questions.jsonl \
    --context llava-bench-in-the-wild/context.jsonl \
    --rule rule.json \
    --answer-list \
        llava-bench-in-the-wild/answers_gpt4.jsonl \
        llava-bench-in-the-wild/answers/$CKPT.jsonl \
    --output llava-bench-in-the-wild/reviews/$CKPT.jsonl

python summarize_gpt_review.py -f llava-bench-in-the-wild/reviews/$CKPT.jsonl

