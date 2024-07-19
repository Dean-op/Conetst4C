import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # 在这里可以进行图像处理，例如绘制矩形框等
        # cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("WebRTC - 打开网络摄像头示例")

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
