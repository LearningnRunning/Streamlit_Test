import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from time import time

save_df = pd.read_csv('result_time.csv')


def display_mono(list,img):
    # Define the points of the polygon
    polygon = np.array(list, dtype=np.int32)
    # Create a mask with the same size as the image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # Draw the polygon on the mask
    cv2.fillPoly(mask, [polygon], 255)
    # Use the mask to cut out the part of the image inside the polygon
    result = cv2.bitwise_and(img, img, mask=mask)

    return result

def display_double(list_l,list_r,img):
    # Define the points of the polygon
    polygon_l = np.array(list_l, dtype=np.int32)
    # Define the points of the polygon
    polygon_r = np.array(list_r, dtype=np.int32)
    # Create a mask with the same size as the image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Draw the polygon on the mask
    cv2.fillPoly(mask, [polygon_l], 255)
    cv2.fillPoly(mask, [polygon_r], 255)

    # Use the mask to cut out the part of the image inside the polygon
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def LandmarkToXY(face_landmarks, landmark, image_width, image_height):
    return min(int(face_landmarks.landmark[landmark].x * image_width), image_width - 1), min(int(face_landmarks.landmark[landmark].y * image_height), image_height - 1)


def XyToList(face_landmarks, landmarks_list, image_width,image_height):
    return [LandmarkToXY(face_landmarks, landmark, image_width, image_height) for landmark in landmarks_list]
    
def LandmarkCalXY_fronthead(idx, face_landmarks, landmarks_list1,landmarks_list2, image_width,image_height):
    x = min(int(face_landmarks.landmark[landmarks_list1[idx]].x * image_width), image_width - 1)
    y = min(int(face_landmarks.landmark[landmarks_list1[idx]].y * image_height), image_height - 1)
    
    x_tmp = min(int(face_landmarks.landmark[landmarks_list2[idx]].x * image_width), image_width - 1)
    y_tmp = min(int(face_landmarks.landmark[landmarks_list2[idx]].y * image_height), image_height - 1)
    
    new_landmark_x = x*2-x_tmp
    new_landmark_y = y*2-y_tmp
    return new_landmark_x, new_landmark_y

def FaceSegmentation(image):
    mp_face_mesh = mp.solutions.face_mesh

    
    # nose mesh numbers
    nose_landmarks = [193, 168, 417, 412, 399, 420, 279, 358, 327, 305, 392, 309, 457, 274, 354, 370, 94, 141, 125, 44, 237, 218, 166, 75, 98, 129, 49, 198, 174, 188, 245]

    # nostrils
    nostrils_landmarks_l = [305, 392, 309, 457, 274, 354, 290]
    # nostrils_landmarks_r = [75,162, 79,227,44,124, 60]

    nostrils_landmarks_r = [125, 44, 237, 59, 240, 60]

    # # dark circles mesh numbers
    # eye_landmarks_l = [46, 53, 52, 65, 55, 193, 188, 174, 47, 100, 118, 117, 127, 162]
    # eye_landmarks_r = [276, 283, 282, 295, 285, 417, 412, 399, 277, 329, 347, 346, 264, 389]

    # # dark circles mesh numbers
    eye_landmarks_l = [46, 53, 52, 65, 55, 193, 245, 188, 174, 188, 174, 198, 49, 129, 98, 165, 186, 216, 207, 187, 116, 127, 162]
    eye_landmarks_r = [276, 283, 282, 295, 285, 417, 412, 399, 420,  279, 358, 327, 391, 410, 436, 427, 376, 345, 264, 389]


    # dark circles mesh numbers
    front_expand_landmarks = [[21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251],[71, 68, 104, 69, 108, 151, 337, 299, 333, 298, 301]]
    front_landmarks = [103 ,54 ,21, 162, 70, 63, 105, 66, 107, 55, 193, 168, 417, 285, 336, 296, 334, 293, 300, 389, 251, 284, 332, 297]

    pore_landmarks = [57, 214, 172,58, 132, 93, 234, 127, 227, 34, *list(range(228,234)), 244, 193, 55, 107, 66, 103, 67, 109, 10, 338, 297, 332, 296, 336, 285, 417, 464, *list(range(453,447,-1)), 264, 447, 366, 401, 435, 397, 434, 287, 269, 267, 0, 37, 39]
    # pore_landmarks = [57, 214, 172,58, 132, 93, 234, 127, 227, 34, *list(range(228,234)), 244, 193, 168, 417, 464, *list(range(453,447,-1)), 264, 447, 366, 401, 435, 397, 434, 287, 269, 267, 0, 37, 39]


    cheek_landmarks = [127, 117, 118, 100, 47, 174, 198, 49, 129, 98, 97, 94, 326, 327, 358, 279, 420, 399, 277, 329, 347, 346, 264, 447, 366, 401, 435, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 177, 137, 227]
    lip_landmarks =[164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 186, 92, 165, 167]
    #lip_landmarks =[0, 267, 269, 270, 409, 291, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]


    iris_landmarks_l = [130, 247, 30,29,27,28,56,190,243, 112, 26, 22, 23, 24, 110, 25]
    iris_landmarks_r = [463, 414, 286, 258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341]


    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:


            # 작업 전에 BGR 이미지를 RGB로 변환합니다.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            annotated_image = image.copy()

            image_width, image_height = annotated_image.shape[1], annotated_image.shape[0]
            face_landmarks = results.multi_face_landmarks[0]
            
            # drawing nose
            nose_list = XyToList(face_landmarks, nose_landmarks, image_width,image_height)
    
            # drawing eyes
            nostrils_list_l = XyToList(face_landmarks, nostrils_landmarks_l, image_width,image_height)
            nostrils_list_r = XyToList(face_landmarks, nostrils_landmarks_r, image_width,image_height)

            # pore line
            pore_list = XyToList(face_landmarks, pore_landmarks, image_width,image_height)
        
            # drawing eyes
            eye_list_l = XyToList(face_landmarks, eye_landmarks_l, image_width,image_height)
            eye_list_r = XyToList(face_landmarks, eye_landmarks_r, image_width,image_height)


            # drawing iris
            iris_list_l = XyToList(face_landmarks, iris_landmarks_l, image_width,image_height)
            iris_list_r = XyToList(face_landmarks, iris_landmarks_r, image_width,image_height)

            # drawing cheek
            cheek_list = XyToList(face_landmarks, cheek_landmarks, image_width,image_height)

            # drawing lip
            lip_list = XyToList(face_landmarks, lip_landmarks, image_width,image_height)
            
            # drawing front head
            front_list = [LandmarkCalXY_fronthead(i, face_landmarks, front_expand_landmarks[0],front_expand_landmarks[1], image_width,image_height) for i in range(len(front_expand_landmarks[0]))]
            front_list_half = XyToList(face_landmarks, front_landmarks, image_width,image_height)

            nose = display_mono(nose_list,annotated_image)

            nostrils = display_double(nostrils_list_l,nostrils_list_r,annotated_image)

            pore = display_mono(pore_list, annotated_image)

            eye = display_double(eye_list_l,eye_list_r,annotated_image)
            iris = display_double(iris_list_l,iris_list_r,annotated_image)
            # Calculate the difference between the two polygons 										    # (subtract poly2 from poly1) 
            only_eye = cv2.subtract(eye, iris)  

            cheek = display_mono(cheek_list,annotated_image)
            lip  = display_mono(lip_list,annotated_image)

            # Calculate the difference between the two polygons 										    # (subtract poly2 from poly1) 
            only_cheek = cv2.subtract(cheek, lip)  
            pore = cv2.subtract(pore, nostrils) 

            only_eye = cv2.subtract(eye, iris) 

            fronthead = display_double(front_list,front_list_half,annotated_image)
            OnlyWrinkle = np.maximum(fronthead, only_eye)

            OnlyFace = np.maximum(fronthead, only_cheek)
            OnlyFace = np.maximum(OnlyFace, only_eye)
            OnlyFace = np.maximum(OnlyFace, nose)

    return OnlyFace ,pore, OnlyWrinkle

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
    
    # alpha = 0.5  # blend factor
    # beta = 0.5   # blend factor
    # gamma = 0    # scalar added to each sum
    # combined = cv2.addWeighted(OnlyFace, alpha, blank_image, beta, gamma)

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

    # alpha = 0.5  # blend factor
    # beta = 0.5   # blend factor
    # gamma = 0    # scalar added to each sum
    # combined = cv2.addWeighted(cropped_img, alpha, result_color, beta, gamma)

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
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
start = time()

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    OnlyFace ,pore, OnlyWrinkle = FaceSegmentation(image)
    sensitive_res = MakeSensitive(OnlyFace)
    pore_res = MakePore(pore)
    wrinkle_res = MakeWrinkle(OnlyWrinkle)


    
    result_time = f"{time() - start:.4f} 秒かかりました。"
    


    alpha = 0.5  # blend factor
    beta = 0.5   # blend factor
    gamma = 0    # scalar added to each sum
    sensitive_combined = cv2.addWeighted(image, alpha, sensitive_res, beta, gamma)
    pore_combined = cv2.addWeighted(image, alpha, pore_res, beta, gamma)
    wrinkle_combined = cv2.addWeighted(image, alpha, wrinkle_res, beta, gamma)

    st.image(sensitive_combined, channels="BGR")
    st.image(pore_combined, channels="BGR")
    # st.image(wrinkle_combined, channels="BGR")

    st.write(result_time)
    print(result_time)
    new_row = pd.DataFrame({"region": [region], "sec": [result_time]})
    save_df = pd.concat([save_df, new_row])
    save_df.to_csv('result_time.csv', encoding='utf-8-sig', index=False)