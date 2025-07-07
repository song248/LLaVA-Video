# LLaVA Video

##  Table of Contents

1. [Model Summary](##model-summary)
2. [Inference](##inference)
3. [Training](##training)
4. [Evaluation](##evaluation-guidance)
6. [Citation](##citation)

## Model Summary

The LLaVA-Video models are 7/72B parameter models trained on [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) and [LLaVA-OneVision Dataset](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data), based on Qwen2 language model with a context window of 32K tokens.

## Train
'''
pip install ninja packaging
pip install flash-attn --no-build-isolation
'''


'''
python llava/train/train.py \
  --data_path custom_exp.yaml \
  --video_folder data \
  --output_dir ./output_llava_video \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1
'''

## Inference

We provide the simple generation process for using our model.  
For more details, you could refer to [Github](https://github.com/LLaVA-VL/LLaVA-NeXT).



## Data Preparation

1. **Download LLaVA-OneVision**  
   Refer to the official instructions here: [LLaVA-OneVision Data](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main/scripts/train#about-the-llava-onevision-data). Make sure to follow the guidelines provided to obtain and organize the data correctly.

2. **Download LLaVA-Video-178K**  
   The dataset is available on Hugging Face: [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K). After downloading, place it in your desired directory.

3. **Update `exp.yaml`**  
   In the [`exp.yaml` file](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/scripts/video/train/exp.yaml), update the file paths to point to the directories where you stored the datasets:
   - **Line 186-Line 263**: Specify the path for the LLaVA-Video-178K dataset.  
   - For other data references, update them to point to your local LLaVA-OneVision data directory.

## Training

[[Scripts]](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/yhzhang/video_dev/scripts/video/train/SO400M_Qwen2_72B_ov_to_video_am9_aug6.sh): Start training models on your single-image/multi-image/video data.


## Evaluation Guidance

We use the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolkit to evaluate our models. Ensure you have installed the LLaVA-NeXT model files as per the instructions in the main README.md.

Install lmms-eval:

> pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

### Reproducing Evaluation Results

Our models' evaluation results can be fully reproduced using the lmms-eval toolkit. After installing lmms-eval and llava, you can run the evaluation using the following commands.

Note: These commands require flash-attn. If you prefer not to install it, disable flash-attn by adding `attn_implementation=None` to the `--model_args` parameter.

Important: Different torch versions may cause slight variations in results. By default in `lmms-eval`, the requirement for torch version is set to the latest version. In `llava` repo, the torch version is set to `2.1.2`. Torch version `2.1.2` would be stable for both `llava` and `lmms-eval`

### Evaluating LLaVA-Video on multiple datasets

We recommend the developers and researchers to thoroughly evaluate the models on more datasets to get a comprehensive understanding of their performance in different scenarios. So we provide a comprehensive list of datasets for evaluation, and welcome to incoporate more evaluation tasks. Please refer to the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for more details.

```bash
# video tasks
accelerate launch --num_processes=8 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average \
--tasks activitynetqa,videochatgpt,nextqa_mc_test,egoschema,video_dc499,videmme,videomme_w_subtitle,perceptiontest_val_mc \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/
```

