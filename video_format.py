import os
import subprocess
from tqdm import tqdm

input_dir = "normal_clip"
output_dir = "normal_clip_mp4"
os.makedirs(output_dir, exist_ok=True)

video_extensions = {'.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv', '.wmv'}
all_files = os.listdir(input_dir)
video_files = [f for f in all_files if os.path.splitext(f)[1].lower() in video_extensions]

print(f"Total {len(video_files)} Video Files to Convert...\n")
for video_file in tqdm(video_files, desc="Converting", unit="file"):
    input_path = os.path.join(input_dir, video_file)
    output_filename = os.path.splitext(video_file)[0] + ".mp4"
    output_path = os.path.join(output_dir, output_filename)

    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-y",
        output_path
    ]

    
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("\n✅ Successfully converted all video files to mp4 format.")
