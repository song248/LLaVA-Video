import os
import argparse
from llava.train.train_mem import main as train_main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, default="data", help="Folder containing videos")
    parser.add_argument("--exp_yaml", type=str, default="custom_exp.yaml", help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default="./output_llava_video")
    parser.add_argument("--model_name_or_path", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov-si")
    parser.add_argument("--vision_tower", type=str, default="google/siglip-so400m-patch14-384")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    os.environ["WANDB_DISABLED"] = "true"

    train_args = [
        "--model_name_or_path", args.model_name_or_path,
        "--version", "qwen_1_5",
        "--data_path", args.exp_yaml,
        "--video_folder", args.video_folder,
        "--vision_tower", args.vision_tower,
        "--mm_projector_type", "mlp2x_gelu",
        "--output_dir", args.output_dir,
        "--num_train_epochs", str(args.epochs),
        "--per_device_train_batch_size", str(args.batch_size),
        "--bf16", "True",
        "--save_steps", "200",
        "--save_total_limit", "1",
        "--learning_rate", "1e-5",
        "--logging_steps", "10",
        "--model_max_length", "32768",
        "--gradient_checkpointing", "True",
        "--lazy_preprocess", "True",
        "--frames_upbound", "64",
        "--add_time_instruction", "True",
        "--force_sample", "True",
        "--report_to", "none"
    ]

    train_main(train_args)

if __name__ == "__main__":
    main()
