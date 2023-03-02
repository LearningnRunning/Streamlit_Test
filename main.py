import streamlit as st
import numpy as np
import cv2
from time import time

from matplotlib import pyplot as plt
def plt_show(img):
    plt.figure(figsize=(10,10))
    plt.axis('off') # 창에있는 x축 y축 제거
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()

st.sidebar.header("Skin diagnosis transmit_receive test in Japan")


st.title("TEST")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
start = time()

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(image, channels="BGR")
    print(f"{time() - start:.4f} sec")
    st.write(f"{time() - start:.4f} 초 소요되었습니다.")