import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
from peft import PeftModel, PeftConfig
from decord import VideoReader, cpu
from PIL import Image
import torchvision.transforms as T

# ─────────────────────────────────────────────
# 설정
video_path = "infer.mp4"
adapter_path = "./work_dirs/llava-lora-fight-vqa-google_siglip-so400m-patch14-384-Qwen_Qwen1.5-1.8B"
base_model = "Qwen/Qwen1.5-1.8B"
vision_tower = "google/siglip-so400m-patch14-384"
device = "cuda" if torch.cuda.is_available() else "cpu"
question = ('Just answer "Yes" or "No". Did this video show any physical violence between people?')
# question = (
#     "Did this video contain any physical altercations or violent incidents between people, "
#     "such as punching, pushing, kicking, threatening gestures, or any other aggressive or hostile behavior?"
# )

# ─────────────────────────────────────────────
# 비디오 → 프레임 추출
def extract_video_frames(path, num_frames=8):
    vr = VideoReader(path, ctx=cpu(0))
    total = len(vr)
    idxs = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = vr.get_batch(idxs).asnumpy()  # (T, H, W, C)
    pil_frames = [Image.fromarray(f) for f in frames]
    return pil_frames

# ─────────────────────────────────────────────
# 모델 로드
print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    trust_remote_code=True
)

# LoRA 연결 (로컬 경로 주의!)
print(f"Loading LoRA adapter from local path: {adapter_path}")
peft_config = PeftConfig.from_pretrained(adapter_path, local_files_only=True)
model = PeftModel.from_pretrained(
    model,
    adapter_path,
    is_trainable=False,
    local_files_only=True
).to(device).eval()

# ─────────────────────────────────────────────
# 이미지 프로세서
processor = AutoImageProcessor.from_pretrained(vision_tower)
image_transform = T.Compose([
    T.Resize((384, 384)),  # SigLIP patch14-384
    T.ToTensor(),
    T.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# ─────────────────────────────────────────────
# 프레임 전처리 및 텍스트 연결
frames = extract_video_frames(video_path, num_frames=8)
frame_tensor = torch.stack([image_transform(f) for f in frames])  # (T, C, H, W)
frame_tensor = frame_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)

# 텍스트 prompt 구성
prompt = f"[Video with {len(frames)} frames]\n{question}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# ─────────────────────────────────────────────
# 텍스트 생성
print("Running inference...")
with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=50,
        do_sample=False
    )

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\n📽️ 질문:", question)
print("🧠 답변:", response)
