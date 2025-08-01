
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle
import math as m
from tkinter import messagebox as ms
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

with open('body_languageLR7000.pkl', 'rb') as f:
    model = pickle.load(f)
print(model)

# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
    degree = int(180/m.pi)*theta
    return degree

cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        # Get height and width of the frame.
        h, w = frame.shape[:2]
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        '''mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )'''
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )
        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        # Export coordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
           
            
           
            # Extract Face landmarks
            #face = results.face_landmarks.landmark
            '''face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            ''' 
            # Concate rows
            row = pose_row #+face_row
            
            '''         #----------------------------angle calculation for feedback----------------------------             
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose()
            lm = results.pose_landmarks
            lmPose  = mp_pose.PoseLandmark
            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
             
            # Right shoulder.
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
             
            # Left ear.
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
 
            # Left hip.
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

            print("--------------------------------angles-----------------------------")
'''           



# Append class name 
#             row.insert(0, class_name)
            
#             # Export to CSV
#             with open('coords.csv', mode='a', newline='') as f:
#                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                 csv_writer.writerow(row) 

            # Make Detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, body_language_prob)
            # Grab ear coords
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [640,480]).astype(int))
            
            cv2.rectangle(image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(body_language_class)*20, coords[1]-30), 
                          (0,0,0), -1)
            cv2.putText(image, body_language_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            '''# Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            #cv2.imshow('Preview', cv2.resize(frame, (1200,600)))
            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)'''
            
            a =  str(round(body_language_prob[np.argmax(body_language_prob)],2))
            print('prb: ',a)
            
            if (body_language_class == 'Tadasana') & (a > "0.95") :
                print("Tadasana")
                #ms.showinfo("Message", "Feedback of Shavasan"+"\n"+"Accuracy :"+str(a))
                colorcode = (0,255,0)
                feedback1 = "Very Good!"
            if (body_language_class == 'Tadasana') & (a <= "0.95") &(a > "0.85")  :
                print("Tadasana")
                #ms.showinfo("Message", "Feedback of Shavasan"+"\n"+"Accuracy :"+str(a))
                colorcode = (0,255,255)
                feedback1 = "Okey,Try Again"
            if (body_language_class == 'Tadasana') & (a <= "0.85") &(a > "0.75"):
                print("Tadasana")
                #ms.showinfo("Message", "Feedback of Shavasan"+"\n"+"Accuracy :"+str(a))
                colorcode = (0,128,255)
                feedback1 = "Medium, Do it correctly!"
            if (body_language_class == 'Tadasana') & (a <= "0.75") &(a > "0.00"):
                print("Tadasana")
                #ms.showinfo("Message", "Feedback of Shavasan"+"\n"+"Accuracy :"+str(a))
                colorcode = (0,0,139)
                feedback1 = "bad Posture, learn it again!"
                
            '''if (body_language_class == 'Tadasana') & (torso_inclination <10):
                   colorcode = (0,0,139)
                   feedback2 = "Good posture"
                if (body_language_class == 'Tadasana') & (torso_inclination >= 10):
                   colorcode = (0,0,139)
                   feedback2 = "Stand streight"'''
#----------------------------------------------------------vajrasan-----------------------------------------------------------               
            if (body_language_class == 'FoldPose') & (a > "0.95") :
               print("FoldPose")
               #ms.showinfo("Message", "Feedback of Vajrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,255,0)
               feedback1 = "Very Good!"
            if (body_language_class == 'FoldPose') & (a <= "0.95") &(a > "0.85")  :
               print("FoldPose")
               #ms.showinfo("Message", "Feedback of Vajrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,255,255)
               feedback1 = "Okey, Try Again"               
            if (body_language_class == 'FoldPose') & (a <= "0.85") &(a > "0.75"):
               print("FoldPose")
               #ms.showinfo("Message", "Feedback of Vajrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,128,255)
               feedback1 = "Medium, Do it correctly!"
            if (body_language_class == 'FoldPose') & (a <= "0.75") &(a > "0.00"):
               print("FoldPose")
               #ms.showinfo("Message", "Feedback of Vajrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,0,139)  
               feedback1 = "bad Posture, learn it again!"
#--------------------------------------------------------------gomukhasan------------------------------------------------
            if (body_language_class == 'Vajrasana') & (a > "0.95"):
               print("Vajrasana")
               #ms.showinfo("Message", "Feedback of Gomukhasana"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,255,0)
               feedback1 = "Very Good!"
            if (body_language_class == 'Vajrasana') & (a <= "0.95") &(a > "0.85") :
               print("Vajrasana")
               #ms.showinfo("Message", "Feedback of Gomukhasana"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,255,255)
               feedback1 = "Okey, Try Again"
            if (body_language_class == 'Vajrasana') & (a <= "0.85") &(a > "0.75"):
               print("Vajrasana")
               #ms.showinfo("Message", "Feedback of Gomukhasana"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,128,255)
               feedback1 = "Medium, Do it correctly!"
            if (body_language_class == 'Vajrasana') & (a <= "0.75") &(a > "0.00"):
               print("Vajrasana")
               #ms.showinfo("Message", "Feedback of Gomukhasana"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,0,139)
               feedback1 = "bad Posture, learn it again!" 
                   
            '''     if (body_language_class == 'Vajrasana') & (neck_inclination > 20) & (neck_inclination < 30) :
                   colorcode = (0,0,139)
                   feedback2 = "Good,Neck is Streight"
                if (body_language_class == 'Vajrasana') & (neck_inclination > 30) & (neck_inclination < 20) :
                   colorcode = (0,0,139)
                   feedback2 = "Adjust your Neck"'''
#--------------------------------------------------------------dhanurasan------------------------------------------------
            if (body_language_class == 'Bhadrasana') & (a > "0.95") :
               print("Bhadrasana")
               #ms.showinfo("Message", "Feedback of Dhanurasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,255,0)
               feedback1 = "Very Good!"
            if (body_language_class == 'Bhadrasana') & (a <= "0.95") &(a > "0.85")  :
               print("Bhadrasana")
               #ms.showinfo("Message", "Feedback of Dhanurasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,255,255)
               feedback1 = "Okey, Try Again"
            if (body_language_class == 'Bhadrasana') & (a <= "0.85") &(a > "0.75"):
               print("Bhadrasana")
               #ms.showinfo("Message", "Feedback of Dhanurasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,128,255)
               feedback1 = "Medium, Do it correctly!"
            if (body_language_class == 'Bhadrasana') & (a <= "0.75") &(a > "0.00"):
               print("Bhadrasana")
               #ms.showinfo("Message", "Feedback of Dhanurasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,0,139) 
               feedback1 = "bad Posture, learn it again!"
#--------------------------------------------------------------bhadrasan------------------------------------------------
            if (body_language_class == 'Dhanurasana') & (a > "0.95") :
               print("Dhanurasana")
               #ms.showinfo("Message", "Feedback of Bhadrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,255,0)
               feedback1 = "Very Good!"
            if (body_language_class == 'Dhanurasana') & (a <= "0.95") &(a > "0.85")  :
               print("Dhanurasana")
               #ms.showinfo("Message", "Feedback of Bhadrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,255,255)
               feedback1 = "Okey, Try Again"
            if (body_language_class == 'Dhanurasana') & (a <= "0.85") &(a > "0.75"):
               print("Dhanurasana")
               #ms.showinfo("Message", "Feedback of Bhadrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,128,255)
               feedback1 = "Medium, Do it correctly!"
            if (body_language_class == 'Dhanurasana') & (a <= "0.75") &(a > "0.00"):
               print("Dhanurasana")
               #ms.showinfo("Message", "Feedback of Bhadrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,0,139)
               feedback1 = "bad Posture, learn it again!"
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------bhadrasan------------------------------------------------
            if (body_language_class == 'Bhujangasan') & (a > "0.95") :
               print("Bhujangasan")
               #ms.showinfo("Message", "Feedback of Bhadrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,255,0)
               feedback1 = "Very Good!"
            if (body_language_class == 'Bhujangasan') & (a <= "0.95") &(a > "0.85")  :
               print("Bhujangasan")
               #ms.showinfo("Message", "Feedback of Bhadrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,255,255)
               feedback1 = "Okey, Try Again"
            if (body_language_class == 'Bhujangasan') & (a <= "0.85") &(a > "0.75"):
               print("Bhujangasan")
               #ms.showinfo("Message", "Feedback of Bhadrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,128,255)
               feedback1 = "Medium, Do it correctly!"
            if (body_language_class == 'Bhujangasan') & (a <= "0.75") &(a > "0.00"):
               print("Bhujangasan")
               #ms.showinfo("Message", "Feedback of Bhadrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,0,139)
               feedback1 = "bad Posture, learn it again!"
                
#--------------------------------------------------------------bhadrasan------------------------------------------------
            if (body_language_class == 'Anjaneyasana') & (a > "0.95") :
               print("Anjaneyasana")
               #ms.showinfo("Message", "Feedback of Bhadrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,255,0)
               feedback1 = "Very Good!"
            if (body_language_class == 'Anjaneyasana') & (a <= "0.95") &(a > "0.85")  :
               print("Anjaneyasana")
               #ms.showinfo("Message", "Feedback of Bhadrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,255,255)
               feedback1 = "Okey, Try Again"
            if (body_language_class == 'Anjaneyasana') & (a <= "0.85") &(a > "0.75"):
               print("Anjaneyasana")
               #ms.showinfo("Message", "Feedback of Bhadrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,128,255)
               feedback1 = "Medium,Do it correctly!"
            if (body_language_class == 'Anjaneyasana') & (a <= "0.75") &(a > "0.00"):
               print("Anjaneyasana")
               #ms.showinfo("Message", "Feedback of Bhadrasan"+"\n"+"Accuracy :"+str(a))
               colorcode = (0,0,139)
               feedback1 = "bad Posture, learn it again!"                
                
                
                
            ''' if (body_language_class == 'Vajarasan') & (a > "0.75") :
                 print("Vajarasan")
                 ms.showinfo("Message", "Feedback of vajrasan \n Regular practice of Vajrasana reduces stress \n and improves concentration."+"\n"+"Accuracy :"+str(a))
                 
            if (body_language_class == 'Gomukhaasan') & (a > "0.75") :
                 print("Vajarasan")
                 ms.showinfo("Message", "Feedback of Gomukhaasan"+"\n"+"Accuracy :"+str(a))'''
                 
                 
            cv2.rectangle(image, (0,0), (950, 60), colorcode, -1)
            #cv2.imshow('Preview', cv2.resize(frame, (1200,600)))
            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            #feedback
            cv2.putText(image, 'Feedback'
                        , (300,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, feedback1
                        , (300,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            ''' cv2.putText(image, 'Improvisation'
                        , (400,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, feedback2
                        , (400,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)'''
            
        except:
            pass
                        
        cv2.imshow('Raw Webcam Feed', cv2.resize(image, (1550,800)))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()