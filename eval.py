import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
from peft import PeftModel, PeftConfig
from decord import VideoReader, cpu
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import precision_score, recall_score, f1_score

# ───── 설정 ─────
video_dir = "eval_video"
adapter_path = "./work_dirs/llava-lora-fight-vqa-google_siglip-so400m-patch14-384-Qwen_Qwen1.5-1.8B"
base_model = "Qwen/Qwen1.5-1.8B"
vision_tower = "google/siglip-so400m-patch14-384"
question = "Did this video show any physical violence between people?"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ───── 모델 로드 ─────
print("🔧 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True).to(device)
peft_config = PeftConfig.from_pretrained(adapter_path, local_files_only=True)
model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False, local_files_only=True).to(device).eval()

# ───── 이미지 전처리 ─────
processor = AutoImageProcessor.from_pretrained(vision_tower)
image_transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# ───── 프레임 추출 함수 ─────
def extract_video_frames(path, num_frames=8):
    try:
        vr = VideoReader(path, ctx=cpu(0))
        total = len(vr)
        idxs = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = vr.get_batch(idxs).asnumpy()
        pil_frames = [Image.fromarray(f) for f in frames]
        return pil_frames
    except Exception as e:
        print(f"⚠️ Failed to process {path}: {e}")
        return None

# ───── 예측 수행 함수 ─────
def run_inference(frames):
    prompt = f"[Video with {len(frames)} frames]\n{question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ───── 평가 루프 ─────
print("🚀 Running evaluation...")
results = []

for fname in os.listdir(video_dir):
    if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue

    fpath = os.path.join(video_dir, fname)
    frames = extract_video_frames(fpath)
    if frames is None:
        continue

    sentence = run_inference(frames).strip().replace("\n", " ")
    predict = 1 if sentence.lower().startswith("yes") else 0
    label = 1 if fname.lower().startswith("fight") else 0

    results.append({
        "name": fname,
        "label": label,
        "predict": predict,
        "sentence": sentence
    })

# ───── 결과 저장 ─────
df = pd.DataFrame(results, columns=["name", "label", "predict", "sentence"])
df.to_csv("eval_results.csv", index=False, encoding="utf-8")
print(f"✅ Evaluation complete. Results saved to eval_results.csv")

# ───── 평가 지표 출력 ─────
print("📈 라벨 분포:\n", df["label"].value_counts())
print("📈 예측 분포:\n", df["predict"].value_counts())

y_true = df["label"]
y_pred = df["predict"]

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n📊 평가 결과:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
