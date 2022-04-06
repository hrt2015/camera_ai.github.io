import cv2
import smtplib
import os
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import datetime
from sendemail import *
import time

label = "Khong Co Dau Hieu Bat Thuong...."
n_time_steps = 10
lm_list = []
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("model.h5")
#Link video cần test
link="D:\\KLTN\\NEW\CLIPTEST\\"

cap = cv2.VideoCapture("D:\\KLTN\\NEW\\CLIPTEST\\binhuongnuoc.mp4")
#cap = cv2.VideoCapture(link+"daubungtriet.mp4")
def sendmail():
    send_email("mingtomdrive1@gmail.com", "Canh Bao Phat Hien Hanh Dong Bat Thuong!!!" + label)


def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0,0,255), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0,255,0)
    thickness = 2
    lineType = 6
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    cv2.putText(img, ("%s %s:%s:%s" % (
        str(datetime.datetime.now().date()), str(datetime.datetime.now().hour), str(datetime.datetime.now().minute),
        str(datetime.datetime.now().second))), (30, 80), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def detect(model, lm_list):

    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)
    print(np.argmax(results[0]))
    mode=np.argmax(results[0])
    if mode == 0:
        label = "VAY TAY"
    elif mode == 1:
        label = "VO TAY"
    elif mode == 2:
        label = "LAC NGUOI"
    elif mode == 3:
        label = "DAU BUNG"

    elif mode == 24:
        label = "DAU BUNG"


    elif mode == 4:
        label = "NHUC DAU"
    elif mode == 25:
        label = "NHUC DAU"
    elif mode == 5:
        label = "DI BO"
    elif mode == 22:
        label = "DI BO"
    elif mode == 6:
        label = "TE XIU"
    elif mode == 7:
        label = "DAU LUNG"
    elif mode == 8:
        label = "DAU CHAN"
    elif mode == 9:
        label = "DAU TAY"
    elif mode == 10:
        label = "DAU CO"
    elif mode == 11:
        label = "UONG NUOC"
    elif mode == 19:
        label = "UONG NUOC"
    elif mode == 23:
        label = "UONG NUOC"
    elif mode == 12:
        label = "TE XIU"
    elif mode == 13:
        label = "VO TAY"
    elif mode == 14:
        label = "DA CHAN"
    elif mode == 15:
        label = "DA CHAN"
    elif mode == 16:
        label = "DUNG DAY"
    elif mode == 17:
        label = "NGHE DIEN THOAI"
    elif mode == 18:
        label = "NGOI XUONG"
    elif mode == 20:
        label = "VAY TAY"
    elif mode == 21:
        label = "VAY TAY"
    elif mode == 26:
        label = "DAU BUNG"
    else:
        label = "Khong Co Dau Hieu Bat Thuong....."
    return label


i = 0
warmup_frames = 0

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i = i + 1
    if i > warmup_frames:
        print("Start detect....")

        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)

            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                # predict
                t1 = threading.Thread(target=detect, args=(model, lm_list,))
                t1.start()
                lm_list = []

            img = draw_landmark_on_image(mpDraw, results, img)

    img = draw_class_on_image(label, img)
    cv2.imshow("TEST ACTION BY CLIP", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
