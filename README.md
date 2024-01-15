# Prometheus-Vision
An Evaluator VLM that is open-source, offers reproducible evaluation, and inexpensive to use. Specifically designed for fine-grained evaluation on customized score rubric, Prometheus-Vision is a good alternative for human evaluation and GPT-4V evaluation.

Our preprint, which describes our method in detail and provides full experimental results and analyses, can be found here:<br>
> [**Prometheus-Vision: Vision-Language Model as a Judge for Fine-Grained Evaluation**](https://arxiv.org/abs/2401.06591).<br>
> [Seongyun Lee](https://www.linkedin.com/in/seongyun-lee-647753233/)$^\ast$, [Seungone Kim](https://seungonekim.github.io/)$^\ast$, [Sue Hyun Park](https://www.linkedin.com/in/sue-hyun-park-2682ba1a0/), [Geewook Kim](https://geewook.kim/), [Minjoon Seo](https://seominjoon.github.io/). Work in progress, arXiv preprint.

## Setup
1. Install package
```bash
conda create -n prometheus-vision python=3.10 -y
conda activate prometheus-vision
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
2. Download data
```
python mmmu_donwload.py
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017
```
## Input and Output Format of Prometheus-Vision
Prometheus-Vision is trained and inferenced using the following input prompt format. Note that you could fill in the instruction, response, reference answer, and score rubrics with your own data.
```text
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, image and a score rubric representing an evaluation criterion is given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{orig_instruction}

###Response to evaluate:
{orig_response}

###Reference Answer (Score 5):
{orig_reference_answer}

###Score Rubrics:
[{orig_criteria}]
Score 1: {orig_score1_description}
Score 2: {orig_score2_description}
Score 3: {orig_score3_description}
Score 4: {orig_score4_description}
Score 5: {orig_score5_description}

###Feedback:
```
You can check the [Perception Collection](https://huggingface.co/datasets/kaist-ai/Perception-Collection) used for training Prometheus-Vision.
## Train
We use [LLaVA](https://github.com/haotian-liu/LLaVA) codebase in developing Prometheus-Vision. Therefore, the following training & inference script is tailored to this. <br>

If you plan to start from a different VLM codebase, you should adapt the format of the data to suit your custom code. <br>

Note that you can also check the data format at [sample_train_data.json](https://github.com/kaistAI/prometheus-vision/blob/main/sample_train_data.json) and [sample_eval_data.json](https://github.com/kaistAI/prometheus-vision/blob/main/sample_train_data.json)
```bash
deepspeed --include llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version plain \
    --data_path TRAINING_DATA_PATH \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```
## Inference
You can place the path of your llava-style model in MODEL_PATH, or insert the trained [Prometheus-Vision](https://huggingface.co/kaist-ai/prometheus-vision-13b-v1.0) we provide.
```bash
python -m llava.eval.model_vqa \
    --model-path MODEL_PATH \
    --question-file EVALUATION_DATA_PATH \
    --answers-file OUTPUT_PATH \
    --temperature 1.0 \
    --top_p 0.9 \
    --conv-mode vicuna_v1
```
Additionally, you can use [Perception-Bench](https://huggingface.co/datasets/kaist-ai/Perception-Bench) when evaluating VLM.

## Citation
If you find our work useful in your work, please consider citing our paper:

```
@misc{lee2024prometheusvision,
      title={Prometheus-Vision: Vision-Language Model as a Judge for Fine-Grained Evaluation}, 
      author={Seongyun Lee and Seungone Kim and Sue Hyun Park and Geewook Kim and Minjoon Seo},
      year={2024},
      eprint={2401.06591},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
