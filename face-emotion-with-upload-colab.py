# %%
# !pip install -q insightface onnxruntime-gpu deepface opencv-python gradio

# %% [markdown]
# # Đây là file chạy upload ảnh thủ công trên co-lab nếu web hết hạn hoặc không đủ GPU

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from deepface import DeepFace
from insightface.app import FaceAnalysis
from PIL import Image

# %%
face_app = FaceAnalysis(allowed_modules=['detection'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# %%
def process_and_plot():
    uploaded = files.upload()
    if not uploaded:
        print("Không có tệp nào được chọn.")
        return

    file_path = list(uploaded.keys())[0]
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = face_app.get(img)
    if not faces:
        print("Không tìm thấy gương mặt nào.")
        return

    largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    bbox = largest_face.bbox.astype(int)
    x1, y1, x2, y2 = bbox

    padding = 40
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(w_img, x2 + padding), min(h_img, y2 + padding)
    cropped_face = img[y1:y2, x1:x2]

    try:
        results = DeepFace.analyze(img_path = cropped_face,
                                   actions = ['emotion'],
                                   enforce_detection = False)
        emotions = results[0]['emotion']
        dominant = results[0]['dominant_emotion']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.imshow(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Detected Face\nDominant: {dominant.upper()}")
        ax1.axis('off')

        emotion_names = list(emotions.keys())
        emotion_values = list(emotions.values())

        indexed_emotions = sorted(zip(emotion_names, emotion_values), key=lambda x: x[1])
        names, values = zip(*indexed_emotions)

        bars = ax2.barh(names, values, color='skyblue')
        ax2.set_xlabel('Tỷ lệ (%)')
        ax2.set_title('Phân tích chi tiết cảm xúc')
        ax2.bar_label(bars, fmt='%.1f%%', padding=3)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Lỗi: {str(e)}")

# %%
process_and_plot()


