# %%
# !pip install -q insightface onnxruntime-gpu deepface opencv-python gradio

# %%
import cv2
import numpy as np
import gradio as gr
from deepface import DeepFace
from insightface.app import FaceAnalysis
from PIL import Image

# %%
# Cài đặt mô hình
face_app = FaceAnalysis(allowed_modules=['detection'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# %%
def analyze_face_emotion(input_img):
    if input_img is None:
        return None, "Vui lòng chọn một hình ảnh!"
    img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)

    faces = face_app.get(img)
    if not faces:
        return None, "Không tìm thấy gương mặt nào trong ảnh."

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
        res = results[0]
        dominant = res['dominant_emotion']
        emotions = res['emotion']
        cropped_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        result_text = f"### Cảm xúc chủ đạo: {dominant.upper()}\n\n"
        sorted_emotions = dict(sorted(emotions.items(), key=lambda item: item[1], reverse=True))
        return cropped_rgb, sorted_emotions
    except Exception as e:
        return None, f"Lỗi phân tích: {str(e)}"

# %%
# Xây dựng giao diện Web (CSS + HTML tích hợp sẵn trong Gradio)
with gr.Blocks(theme=gr.themes.Soft(), title="AI Emotion Detector") as demo:
    gr.Markdown("""
    # Hệ Thống Phân Tích Cảm Xúc
    Tải ảnh lên để hệ thống tự động nhận diện khuôn mặt và phân tích trạng thái cảm xúc.
    """)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Ảnh gốc", type="numpy")
            with gr.Row():
                btn_reset = gr.Button("Làm mới", variant="secondary")
                btn_analyze = gr.Button("Phân Tích Cảm Xúc", variant="primary")
        with gr.Column():
            output_face = gr.Image(label="Khuôn mặt đã nhận diện")
            output_chart = gr.Label(label="Tỷ lệ cảm xúc (%)", num_top_classes=5)
    # Xử lý sự kiện
    btn_analyze.click(fn=analyze_face_emotion,
                      inputs=input_image,
                      outputs=[output_face, output_chart])
    btn_reset.click(fn=lambda: [None, None, None],
                    inputs=None,
                    outputs=[input_image, output_face, output_chart])

# %%
# Run web
demo.launch(share=True, debug=True)


