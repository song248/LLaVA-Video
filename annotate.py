import os
import json

# 설정
DATA_DIR = "data"
QUESTION = "Did this video contain any physical altercations or violent incidents between people,\
 such as punching, pushing, kicking, threatening gestures, or any other aggressive or hostile behavior?"
OUTPUT_JSON = "labeled.json"

# 초기화
dataset = []

# violence, normal 폴더 탐색
for label_folder in ["normal", "violence"]:
    folder_path = os.path.join(DATA_DIR, label_folder)
    if not os.path.exists(folder_path):
        continue

    # 라벨 설정
    answer = "Yes" if label_folder == "violence" else "No"

    # 영상 파일 순회
    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith((".mp4", ".webm", ".avi", ".mov", ".mkv")):
            continue

        file_path = os.path.join(folder_path, file_name)

        # 항목 추가
        dataset.append({
            "video_path": file_path,
            "question": QUESTION,
            "answer": answer
        })

# 저장
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"총 {len(dataset)}개의 라벨이 '{OUTPUT_JSON}' 파일에 저장되었습니다.")
