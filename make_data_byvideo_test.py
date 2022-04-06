import cv2
import mediapipe as mp
import pandas as pd
import datetime
import time

# Đọc ảnh từ webcam
cap = cv2.VideoCapture("D:\\KLTN\\NEW\\DATASET\\a04\\VideoFull.mp4")

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "TESTTRAINUONGNUOC500"
no_of_frames =500

def make_landmark_timestep(results):
    print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    # Vẽ các đường nối
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    cv2.putText(img, ("%s %s:%s:%s" % (
        str(datetime.datetime.now().date()), str(datetime.datetime.now().hour), str(datetime.datetime.now().minute),
        str(datetime.datetime.now().second))), (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return img


while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    if ret:
        # Nhận diện pose
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            # Ghi nhận thông số khung xương
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            # Vẽ khung xương lên ảnh
            frame = draw_landmark_on_image(mpDraw, results, frame)

        cv2.imshow("THU THAP DU LIEU", frame)
        if cv2.waitKey(1) == ord('q'):
            break

# Write vào file csv
df  = pd.DataFrame(lm_list)
link= 'D:\\KLTN\\NEW\\ACTION_DATASET_CSV\\'
df.to_csv(link+label + ".txt")
cap.release()
cv2.destroyAllWindows()