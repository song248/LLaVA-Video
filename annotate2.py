import json

with open("labeled.json", "r") as f:
    data = json.load(f)

converted = []

for item in data:
    video_path = item["video_path"]

    # 기존처럼 "data/" prefix 제거
    if video_path.startswith("data/"):
        video_path = video_path[len("data/"):]

    converted.append({
        "video": video_path,
        "conversations": [
            {
                "from": "human",
                "value": 'Is there any violence between people in this video? Answer "Yes" or "No" only.'
            },
            {
                "from": "gpt",
                "value": item["answer"]
            }
        ]
    })

with open("labeled_converted.json", "w") as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

print("labeled_converted.json 생성됨.")
