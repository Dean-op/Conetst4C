def predictRealtime(model):
    st.subheader("实时监测")
    run = st.button("运行")
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("未能获取视频流")
            break
        processed_frame = process_frame(frame, model)
        stframe.image(processed_frame, channels="BGR", use_column_width=True)

    cap.release()