from ultralytics import YOLO
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import tempfile


def predictVideo(uploaded_file, model):
    with st.spinner("视频处理中..."):
        if uploaded_file is not None:
            temp_file_path = os.path.join(tempfile.gettempdir(), "input_video.mp4")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())

            video = cv2.VideoCapture(temp_file_path)

            frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video.get(cv2.CAP_PROP_FPS))
            # frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter("output_video.mp4", fourcc, fps, (frame_width, frame_height))

            while True:
                ret, frame = video.read()
                if not ret:
                    break

                processed_frame = process_frame(frame, model)
                output_video.write(processed_frame)

            video.release()
            output_video.release()
            st.success("视频处理完成,点击下载按钮保存本地:")
            st.download_button(label="下载视频", data=open("output_video.mp4", "rb").read(),
                               file_name="output_video.mp4")


def process_frame(frame, model):
    pred = model.predict(frame)[0].plot()
    return pred


def predictImage(img, model):
    image = Image.open(img)
    img_array = np.array(image)
    results = model.predict(img_array)
    pred = model.predict(img_array)[0].plot()
    return results, pred


def main():
    with st.sidebar:
        st.title("About:")
        st.markdown(
            "- 军用战斗机识别检测系统\n" \
            "- 作品编号：2024019044\n" \
            # "- "
        )

    st.title("军用战斗机识别检测系统")
    path = "2024019044项目/runs/detect/train/weights/best.pt"
    my_model = YOLO(path)
    img_file_buffer = st.file_uploader('上传图像(jpg、jpeg、 png、 gif)或视频(mp4)', type=["jpg", "jpeg", "png", "gif", "mp4"])
    button = st.button("提交")

    if button:
        # st.snow()
        if img_file_buffer is None:
            st.error("❌请上传图片('jpg','jpeg','png','gif')或图像(mp4)")
        else:
            mime_type = img_file_buffer.type
            if "image" in mime_type:
                results, pred = predictImage(img_file_buffer, my_model)
                st.image(pred, width=550, channels="RGB")
                for result in results:
                    if result.boxes:
                        box = result.boxes[0]
                        class_id = int(box.cls)
                        object_name = my_model.names[class_id]
                        st.write("战斗机型号:", object_name)

            elif "video" in mime_type:
                predictVideo(img_file_buffer, my_model)


if __name__ == "__main__":
    main()
