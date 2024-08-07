import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO
import mysql.connector
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

st.title("军用战斗机型号识别系统")
st.sidebar.text("军用战斗机识别检测系统")
st.sidebar.text("作品编号:2024019044")
st.sidebar.text("——————————————————————————————————")

st.sidebar.title("上传图像或视频")
upload_mode = st.sidebar.selectbox("选择上传模式", ["图像", "视频", "实时监测"])


def connectDB():
    try:
        mydb = mysql.connector.connect(
            host="sql12.freemysqlhosting.net",
            user="sql12720854",
            password="TYsA4pGcua",
            database="sql12720854"
        )
        return mydb
    except mysql.connector.Error as err:
        st.error(f"数据库连接失败: {err}")
        return None


def dispFighterInfo(mydb, name):
    myCursor = mydb.cursor(dictionary=True)
    sql = "SELECT * FROM fighter_info WHERE 名称 = %s"
    myCursor.execute(sql, (name,))
    result = myCursor.fetchall()
    myCursor.close()

    if result:
        st.write("详细信息:")
        st.table(result)
    else:
        st.write("未找到该战斗机的信息")


def predictImage(img, model, conf_threshold=0.8):
    image = Image.open(img)
    img_array = np.array(image)
    results = model.predict(img_array, conf=conf_threshold)

    if not results or not results[0].boxes:
        st.error("无法确定该战斗机信息")
        return None, None

    pred_img = results[0].plot()
    return results, pred_img


def process_frame(frame, model):
    results = model.predict(frame)
    if not results or not results[0].boxes:
        return frame
    pred_img = results[0].plot()
    return pred_img


def predictVideo(uploaded_file, model):
    with st.spinner("检测目标中..."):
        temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        video = cv2.VideoCapture(temp_file_path)
        stframe = st.empty()

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = process_frame(frame, model)
            stframe.image(processed_frame, use_column_width=True)

        video.release()


class VideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(img)
        if results and results[0].boxes:
            img = results[0].plot()
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    mydb = connectDB()
    if mydb is None:
        return

    model = YOLO('2024019044项目/runs/detect/train/weights/best.pt')

    if upload_mode == "图像":
        uploaded_file = st.sidebar.file_uploader("上传图像", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.sidebar.image(uploaded_file, caption="上传的图像", use_column_width=True)
            results, pred_img = predictImage(uploaded_file, model)
            if pred_img is not None:
                st.image(pred_img, caption="识别结果：")
                if results:
                    for result in results:
                        if result.boxes:
                            box = result.boxes[0]
                            conf = box.conf[0].item()
                            if conf >= 0.81:
                                class_id = int(box.cls)
                                object_name = model.names[class_id]
                                st.markdown("<h4 style='color: black;'>战斗机信息:</h4>", unsafe_allow_html=True)
                                st.subheader(object_name)
                                dispFighterInfo(mydb, object_name)
                            else:
                                st.markdown("<h4 style='color: black;'>无法确定该战斗机信息</h4>",
                                            unsafe_allow_html=True)

    elif upload_mode == "视频":
        uploaded_file = st.sidebar.file_uploader("上传视频", type=["mp4", "avi"])
        if uploaded_file is not None:
            st.sidebar.text("上传的视频:")
            st.sidebar.video(uploaded_file)
            start = st.button("开始")
            if start:
                predictVideo(uploaded_file, model)

    elif upload_mode == "实时监测":
        st.subheader("实时监测:")
        rtc_config = {
            "iceServers": [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:stun1.l.google.com:19302"},
                {"urls": "stun:stun2.l.google.com:19302"},
                {"urls": "stun:stun3.l.google.com:19302"},
                {"urls": "stun:stun4.l.google.com:19302"},
                # Add other STUN servers from the valid_ipv4s.txt file
                {"urls": "stun:stun.ekiga.net"},
                {"urls": "stun:stun.ideasip.com"},
                {"urls": "stun:stun.schlund.de"},
                {"urls": "stun:stun.stunprotocol.org:3478"},
                {"urls": "stun:stun.voiparound.com"},
                {"urls": "stun:stun.voipbuster.com"},
                {"urls": "stun:stun.voipstunt.com"},
                {"urls": "stun:stun.voxgratia.org"},
                {"urls": "stun:stun.services.mozilla.com"},
                {
                    "urls": "turn:TURN_SERVER_URL",
                    "username": "USERNAME",
                    "credential": "CREDENTIAL"
                }
            ]
        }
        webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: VideoProcessor(model),
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
        )


if __name__ == "__main__":
    main()
