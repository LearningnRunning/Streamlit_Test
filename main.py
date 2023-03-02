import streamlit as st
import cv2
import numpy as np
import pandas as pd

from time import time

save_df = pd.read_csv('result_time.csv')

st.sidebar.header("所要時間テスト")


st.title("TEST")
st.write("画像送受信にかかる時間を計算しようとするテストです。 画像をアップロードしていただくと時間が表示されます。")

region = st.text_input("검색할 지역을 입력해주세요(ex 영등포구 or 속초시)", value="서울특별시 중구")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
start = time()

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    # OnlyFace ,pore, OnlyWrinkle = FaceSegmentation(image)
    # result = MakePore(OnlyFace)
    st.image(image, channels="BGR")
    result_time = f"{time() - start:.4f} sec"
    st.write(result_time)
    new_row = pd.DataFrame({"region": [region], "sec": [result_time]})
    save_df = pd.concat([save_df, new_row])
    save_df.to_csv('result_time.csv', encoding='utf-8-sig', index=False)

    