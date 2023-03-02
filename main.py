import streamlit as st
import cv2
import numpy as np

from time import time


st.sidebar.header("所要時間テスト")


st.title("TEST")
st.write("画像送受信にかかる時間を計算しようとするテストです。 画像をアップロードしていただくと時間が表示されます。")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
start = time()

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    # OnlyFace ,pore, OnlyWrinkle = FaceSegmentation(image)
    # result = MakePore(OnlyFace)
    st.image(image, channels="BGR")
    print(f"{time() - start:.4f} sec")
    st.write(f"{time() - start:.4f} 秒かかりました。")