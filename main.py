import streamlit as st

import numpy as np
import pandas as pd
import cv2
from google.cloud import firestore
from time import time
from datetime import datetime
from PIL import Image
import io
from google.oauth2 import service_account
import json

save_df = pd.read_csv('result_time.csv')


def report(region, img, result_time):
    # Authenticate to Firestore with the JSON account key.
    key_dict = json.loads(st.secrets['textkey'])
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds)
    result_time = f"{time() - start:.4f}"
    data = {
        'region': region,
        'img': img,
        'datetime': datetime.now(),
        'time' : result_time
        }
    # Create a reference to the Google post.
    doc_ref = db.collection('test_img').add(data)

    return doc_ref[1].id, db, result_time


def retrieve(id, db):
    doc_ref = db.collection('test_img').document(id)
    doc = doc_ref.get().to_dict()
    image_bytes = doc['img']
    return image_bytes


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

    return blank_image

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

    return result_color

def MakeWrinkle(OnlyWrinkle):
    gray = cv2.cvtColor(OnlyWrinkle, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (23, 23), 0)

    canny = cv2.Canny(blurred, 12, 22)

    _, thresh = cv2.threshold(gray, 10, 200, cv2.THRESH_BINARY)
    canny_th = cv2.Canny(thresh, 1, 3)
    # Create a kernel for dilation
    kernel = np.ones((7,7), np.uint8)
    # Dilate the edges to make them thicker
    thicker = 5
    dilated_edges = cv2.dilate(canny_th, kernel, iterations=thicker)

    # plt_show(dilated_edges)
    result = cv2.subtract(canny, dilated_edges) 
    # Create a kernel for dilation
    kernel = np.ones((3,3), np.uint8)
    # Dilate the edges to make them thicker
    thicker = 2
    dilated_result= cv2.dilate(result, kernel, iterations=thicker)
    result_color = cv2.cvtColor(dilated_result, cv2.COLOR_GRAY2BGR)
    # Set the color of the edges to red
    result_color[dilated_result != 0] = [24, 133, 255]
    
    return result_color

st.sidebar.header("所要時間テスト")


st.title("TEST")
st.write("画像送受信にかかる時間を計算しようとするテストです。 画像をアップロードしていただくと時間が表示されます。")

region = st.text_input("現在の地域を入力してください。 （例:東京市）", value="東京市")
uploaded_file = st.file_uploader("画像を入力してください。", type=["jpg", "jpeg", "png"])

alpha = 0.5  # blend factor
beta = 0.5   # blend factor
gamma = 0    # scalar added to each sum

if uploaded_file is not None:
    start = time()
    # assume `uploaded_file` is the image uploaded from Streamlit
    image = Image.open(uploaded_file)

    # resize the image to (1862, 4032)
    resized_image = image.resize((1862, 4032))

    # convert the image to byte type
    buffered = io.BytesIO()
    resized_image.save(buffered, format="JPEG")
    byte_image = buffered.getvalue()


    # img = retrieve(id, db)
    # f1 = FireStore(region, uploaded_file.read())
    # img = f1.retrieve()

    # file_bytes = np.asarray(bytearray(resized_image), dtype=np.uint8)
    np_image = np.array(resized_image)

    # convert the NumPy array to an OpenCV image format
    image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    sensitive_res = MakeSensitive(image)
    pore_res = MakePore(image)
    

    sensitive_combined = cv2.addWeighted(image, alpha, sensitive_res, beta, gamma)
    pore_combined = cv2.addWeighted(image, alpha, pore_res, beta, gamma)


    st.image(sensitive_combined, channels="BGR")
    st.image(pore_combined, channels="BGR")
    # st.image(wrinkle_combined, channels="BGR")

    
    
    id, db, result_time = report(region, byte_image, start)
    
    st.write(result_time + "秒かかりました。")
    new_row = pd.DataFrame({"region": [region], "sec": [result_time]})
    save_df = pd.concat([save_df, new_row])
    save_df.to_csv('result_time.csv', encoding='utf-8-sig', index=False)