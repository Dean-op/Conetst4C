from ultralytics import YOLO
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import tempfile


def predictVideo(uploaded_file, model):
    # 一个简单的视频处理动画
    with st.spinner("视频处理中..."):
        if uploaded_file is not None:
            # 将上传的视频文件保存到临时目录
            temp_file_path = os.path.join(tempfile.gettempdir(), "input_video.mp4")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())
            # 读取上传的视频文件
            video = cv2.VideoCapture(temp_file_path)

            # 获取视频参数：帧宽、帧高、帧率
            frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video.get(cv2.CAP_PROP_FPS))

            # 设置输出视频文件参数
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter("output_video.mp4", fourcc, fps, (frame_width, frame_height))

            # 逐帧处理视频并写入输出视频文件
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                # 调用模型处理视频帧
                processed_frame = process_frame(frame, model)
                output_video.write(processed_frame)

            # 释放视频资源
            video.release()
            output_video.release()

            # 显示视频处理完成的提示信息，并提供下载按钮
            st.success("视频处理完成,点击下载按钮保存本地:")
            st.download_button(label="下载视频", data=open("output_video.mp4", "rb").read(),
                               file_name="output_video.mp4")


def process_frame(frame, model):
    # 使用模型对视频帧进行预测并绘制结果
    pred = model.predict(frame)[0].plot()
    return pred


def predictImage(img, model):
    image = Image.open(img)
    # 将图像转换为numpy数组
    img_array = np.array(image)
    # 使用模型对图像进行预测
    results = model.predict(img_array)
    # 从预测结果中提取第一个结果，并绘制预测边界框
    pred = model.predict(img_array)[0].plot()
    # 返回识别结果和绘制了预测边界框的图像对象
    return results, pred


def main():
    # 加载模型
    path = "2024019044项目/runs/detect/train/weights/best.pt"
    my_model = YOLO(path)

    #页面标题说明等
    with st.sidebar:
        st.title("About:")
        st.markdown(
            "- 基于长河算法可视化开发平台实现军用战斗机识别检测系统\n" \
            "- 作品编号：2024019044\n" \
            # "- "
        )
    st.title("军用战斗机识别检测系统")
    st.write("支持以下九种战斗机型号：")
    st.write("(E2、J20、B2、F14、Tornado、F4、B52、JAS39、Mirage2000)")

    #上传图像/视频
    img_file_buffer = st.file_uploader('上传图像(jpg、jpeg、 png、 gif)或视频(mp4)',
                                       type=["jpg", "jpeg", "png", "gif", "mp4"])
    button = st.button("提交")

    if button:
        #对上传文件格式合法性判断
        if img_file_buffer is None:
            st.error("❌请上传图片('jpg','jpeg','png','gif')或图像(mp4)")
        else:
            #获取文件类型
            mime_type = img_file_buffer.type
            if "image" in mime_type:
                #使用模型对图像进行预测并显示预测结果图像
                results, pred = predictImage(img_file_buffer, my_model)
                st.image(pred, width=550, channels="RGB")

                for result in results:
                    # 检查预测结果是否包含边界框信息
                    if result.boxes:
                        box = result.boxes[0]

                        # 获取类别 ID映射到对应的型号并显示到界面上
                        class_id = int(box.cls)
                        object_name = my_model.names[class_id]
                        st.write("战斗机型号:", object_name)

            elif "video" in mime_type:
                #视频处理
                predictVideo(img_file_buffer, my_model)


if __name__ == "__main__":
    main()
