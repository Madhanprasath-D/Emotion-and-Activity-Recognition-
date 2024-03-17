import numpy as np
import csv
import os
import mediapipe as mp
import cv2

m_draw = mp.solutions.drawing_utils
m_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

c_name ="Happy"

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
            # print(p_row)
            face = res.face_landmarks.landmark
            f_row = list(np.array([[i.x, i.y, i.z, i.visibility] for i in face]).flatten())
            # print(f_row)

            row = p_row+f_row
            row.insert(0,c_name)
            print(row)

            with open('data.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            


        except Exception as err:
            print(err)

        
        cv2.imshow("MY Application",img)
        cv2.waitKey(1)

    
cap.release()
cv2.destroyAllWindows()


# num_coords = len(res.pose_landmarks.landmark)+len(res.face_landmarks.landmark)

# land_marks = ['class']
# for i in range(1,502):
#     land_marks += ['x{}'.format(i), 'y{}'.format(i), 'z{}'.format(i), 'v{}'.format(i)]

# # print(land_marks)

# with open('data.csv', mode='w', newline='') as f:
#     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
#     csv_writer.writerow(land_marks)