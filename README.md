# VisATB

Official implementation of "*Adaptive Task Balancing for Visual Instruction Tuning via Inter-Task Contribution and Intra-Task Difficulty*", which has been accepted by WWW 2026.
Paper Link: https://arxiv.org/abs/2403.04343.

## Introduction

Visual instruction tuning is a key training stage of large multimodal models. However, when learning multiple visual tasks simultaneously, this approach often results in suboptimal and imbalanced overall performance due to latent knowledge conflicts across tasks. To mitigate this issue, we propose a novel **A**daptive **T**ask **B**alancing approach tailored for **vis**ual instruction tuning (**VisATB**). Specifically, we measure two critical dimensions for visual task balancing based on validation performance: (1) *Inter-Task Contribution*, the mechanism where learning one task enhances the performance on others owing to shared knowledge across tasks, and (2) *Intra-Task Difficulty*, which denotes the inherent learning difficulty of a single task. Furthermore, we propose prioritizing three categories of tasks with greater weight: those that offer substantial contributions to others, those that receive minimal contributions from others, and those that present high learning difficulties. Among these three task weighting strategies, the first and third focus on improving overall performance, and the second targets the mitigation of performance imbalance. Extensive experiments on three benchmarks demonstrate that our VisATB approach consistently achieves superior and more balanced overall performance in visual instruction tuning.

![](assets/overall.jpg)

Comparative results on the Academic Benchmark:

![](assets/results.jpg)

## Setup environment

```
git clone https://github.com/YanqiDai/VisATB.git
cd VisATB
conda create -n visatb python=3.10
conda activate visatb
pip install -r requirements.txt
```

Note that you should ensure that `transformers` version is `4.31.0` to avoid potential bugs. Then you need to replace the `transformers/trainer.py` and `transformers/models/llama/modeling_llama.py` files with the provided modified versions in the `VisATB/transformers/` folder.

The core modification introduces a new `weighter` variable that enables adaptive task balancing during training. To understand the implementation details, simply search for the `weighter` keyword across the modified files. This serves as a valuable reference for integrating VisATB into your own codebase.

## Data Preparation

Download the annotation files from [Modelscope](https://modelscope.cn/datasets/YechielDai/VisATB) and place them in the `data/` folder. Download the images from [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning), [ShareGPT4V](https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/docs/Data.md#prepare-images) and [M3IT](https://huggingface.co/datasets/MMInstruction/M3IT), and place them in the `data/images/` folder.

## Training

The training scripts are provided in the `scripts_visatb/` folder.

The first step of VisATB is to training multiple models with different sub datasets, as detailed in the paper. You can run the following command to train these models:

```
# DATA_PATH="path/to/sub_dataset"
# IMAGE_FOLDER="path/to/images"
# VISATB_WEIGHTS="1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0"
# OUTPUT_DIR="path/to/output_dir"
bash scripts_visatb/finetune_academic.sh
```

Then, you need to evaluate these models on all validation sets. You can run the following command to evaluate a model:

```
python inference_visatb.py 
    --model_path "path/to/model_checkpoint" \
    --image_folder "path/to/images" \
    --question_file "path/to/validation_file" \
    --answers-folder "path/to/save_answers"

# for caption tasks, such as ShareGPT4V
python eval/caption.py -a "path/to/answer_file"
# for caption list tasks, such as RefCOCO-caption 
python eval/caption_list.py -a "path/to/answer_file"
# for grounding tasks, such as RefCOCO-bbox
python eval/grounding.py -a "path/to/answer_file"
# for VQA tasks, such as GQA
python eval/qa.py -a "path/to/answer_file"
```

After obtaining the validation results, you can calculate inter-task contribution and intra-task difficulty, and use the provided `weighting_visatb.py` script to compute the final task weights:

```
python weighting_visatb.py
```

Finally, you can use the computed task weights to finetune the final model with VisATB:

```
# DATA_PATH="path/to/entire_dataset"
# IMAGE_FOLDER="path/to/images"
# VISATB_WEIGHTS="computed_task_weights"
# OUTPUT_DIR="path/to/output_dir"
bash scripts_visatb/finetune_academic.sh
```

## Evaluation

The evaluation process is similar to [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md). The scripts are provided in the `scripts_visatb/eval` folder.



