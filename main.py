import numpy as np
import csv
import os
import mediapipe as mp
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle


with open('model.pkl', mode='rb') as f:
    model = pickle.load(f)

# print(model)

m_draw = mp.solutions.drawing_utils
m_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

red_color = (255,0,0)
normal_color = (80,110,10)

with m_holistic.Holistic(min_tracking_confidence=0.5,min_detection_confidence=0.5) as holistic:

    while True:

        sucess , img = cap.read()
        cv2.imshow("MY APPLICATTION",img)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res = holistic.process(imgRGB)

        m_draw.draw_landmarks(img,res.face_landmarks,m_holistic.FACEMESH_CONTOURS,
                            m_draw.DrawingSpec(color = (80,110,10), thickness=1,circle_radius=1),
                            m_draw.DrawingSpec(color = (80,256,121),thickness=1,circle_radius=1)
        )

        m_draw.draw_landmarks(img,res.left_hand_landmarks,m_holistic.HAND_CONNECTIONS,
                            m_draw.DrawingSpec(color = (80,25,10), thickness=2,circle_radius=3),
                            m_draw.DrawingSpec(color = (80,45,121),thickness=2,circle_radius=2)
        )

        m_draw.draw_landmarks(img,res.right_hand_landmarks,m_holistic.HAND_CONNECTIONS,
                            m_draw.DrawingSpec(color = (121,22,75), thickness=2,circle_radius=3),
                            m_draw.DrawingSpec(color = (121,45,250),thickness=2,circle_radius=2)
        )

        m_draw.draw_landmarks(img, res.pose_landmarks,m_holistic.POSE_CONNECTIONS,
                            m_draw.DrawingSpec(color = (245,117,66), thickness=2,circle_radius=2),
                            m_draw.DrawingSpec(color = (245,66,230),thickness=2,circle_radius=2)
        )

        try:
            pose = res.pose_landmarks.landmark
            p_row = list(np.array([[i.x, i.y, i.z, i.visibility] for i in pose]).flatten())
            face = res.face_landmarks.landmark
            f_row = list(np.array([[i.x, i.y, i.z, i.visibility] for i in face]).flatten())

            row = p_row+f_row
            
            x = pd.DataFrame([row])
            output = model.predict(x)[0]
            # # output_prob = model.predict_proba(data)[0]
            print(output)
            
            # points = tuple(np.multiply(
            #     np.array((
            #         res.pose_landmarks[m_holistic.PoseLandmark.LEFT_EAR].x,
            #         res.pose_landmarks[m_holistic.PoseLandmark.LEFT_EAR].y)
            #     ), [640,480]
            # ).astype(int))

            # print(points)
            
            # cv2.rectangle(img,
            #     (points[0], points[1]+5),
            #     (points[0]+len(output)*20,points[1]-30),
            #     (245,117,16),-1)
            
            # cv2.putText(img,
            #     output, points,
            #     cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA
            # )
            
            print("compite")

        except Exception as err:
            print(err)

        
        cv2.imshow("MY Application",img)
        cv2.waitKey(1)

    
cap.release()
cv2.destroyAllWindows()