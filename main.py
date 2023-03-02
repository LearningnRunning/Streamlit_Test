import streamlit as st
import cv2
import numpy as np
import pandas as pd

from time import time

save_df = pd.read_csv('result_time.csv')

def MakeSensitive(OnlyFace):
    # Split image into BGR channels
    b, g, r = cv2.split(OnlyFace)
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(OnlyFace, cv2.COLOR_BGR2HSV)

    # Split image into Hue, Saturation, and Value channels
    h, s, v = cv2.split(hsv_image)

    # Increase saturation of red channel by a factor of 1.5
    s_adjusted = np.where((r > b) & (r > g), np.clip(s * 2, 0, 255), s).astype(np.uint8)

    # Merge adjusted channels back into image
    hsv = cv2.merge([h, s_adjusted, v])

    # Define a range of red color in HSV
    lower_red = np.array([0,100, 100])
    upper_red = np.array([3,255,255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blank_image = np.zeros(OnlyFace.shape, dtype=np.uint8)
    cv2.drawContours(blank_image, contours, -1, (150, 255, 0), 3, cv2.LINE_8, hierarchy)
    
    alpha = 0.5  # blend factor
    beta = 0.5   # blend factor
    gamma = 0    # scalar added to each sum
    combined = cv2.addWeighted(OnlyFace, alpha, blank_image, beta, gamma)

    return combined


st.sidebar.header("所要時間テスト")


st.title("TEST")
st.write("画像送受信にかかる時間を計算しようとするテストです。 画像をアップロードしていただくと時間が表示されます。")

region = st.text_input("現在の地域を入力してください。 （例:東京市）", value="東京市")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
start = time()

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    result = MakeSensitive(image)
    # OnlyFace ,pore, OnlyWrinkle = FaceSegmentation(image)
    # result = MakePore(OnlyFace)
    result

    st.image(result, channels="BGR")
    result_time = f"{time() - start:.4f} 秒かかりました。"
    st.write(result_time)
    print(result_time)
    new_row = pd.DataFrame({"region": [region], "sec": [result_time]})
    save_df = pd.concat([save_df, new_row])
    save_df.to_csv('result_time.csv', encoding='utf-8-sig', index=False)

    