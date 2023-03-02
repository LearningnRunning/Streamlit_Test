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

def MakePore(cropped_img):
    lab = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=6.5,tileGridSize=(15, 15))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))


    cont_dst = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(cont_dst, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (23, 23), 0)


    canny = cv2.Canny(blurred, 40, 60)

    _, thresh = cv2.threshold(blurred, 30, 200, cv2.THRESH_BINARY)

    canny_th = cv2.Canny(thresh, 1, 3)

    # Create a kernel for dilation
    kernel = np.ones((7,7), np.uint8)
    # Dilate the edges to make them thicker
    thicker = 7
    dilated_edges = cv2.dilate(canny_th, kernel, iterations=thicker)

    result = cv2.subtract(canny, dilated_edges) 
    # Create a kernel for dilation
    kernel = np.ones((3,3), np.uint8)
    # Dilate the edges to make them thicker
    thicker = 2
    dilated_result= cv2.dilate(result, kernel, iterations=thicker)
    result_color = cv2.cvtColor(dilated_result, cv2.COLOR_GRAY2BGR)
    # Set the color of the edges to red
    result_color[dilated_result != 0] = [24, 133, 255]

    alpha = 0.5  # blend factor
    beta = 0.5   # blend factor
    gamma = 0    # scalar added to each sum
    combined = cv2.addWeighted(cropped_img, alpha, result_color, beta, gamma)

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
    sensitive_res = MakeSensitive(image)
    pore_res = MakePore(image)
    # OnlyFace ,pore, OnlyWrinkle = FaceSegmentation(image)
    # result = MakePore(OnlyFace)

    st.image(sensitive_res, channels="BGR")
    st.image(pore_res, channels="BGR")
    result_time = f"{time() - start:.4f} 秒かかりました。"
    st.write(result_time)
    print(result_time)
    new_row = pd.DataFrame({"region": [region], "sec": [result_time]})
    save_df = pd.concat([save_df, new_row])
    save_df.to_csv('result_time.csv', encoding='utf-8-sig', index=False)

    