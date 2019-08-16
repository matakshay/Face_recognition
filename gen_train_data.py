#Generates Training Data
#Author: Akshay Mattoo

import cv2 #Import OpenCV library
import numpy as np #Import Numpy library

#Initialize to capture video from the delfault web camera
cap = cv2.VideoCapture(0)

#Detect faces using Haar cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
#intitilze a list to store the flattened image
face_data = []
#Set the path to the destination where above image will be stored
dataset_path = './train_data'
#Input the name of the person whose image is to be detected and saved
file_name = input("Enter the name of the person: ")

while True:
    #Store the caputred frame
    ret, frame = cap.read()

    #If frame wasn't captured, try again
    if ret==False:
        continue


    faces = face_cascade.detectMultiScale(frame,1.3,5)
    if len(faces)==0:
        continue
    
    #Sort the faces on the basis of their sizes
    faces = sorted(faces, key = lambda f:f[2]*f[3]) #Since faces is a tuple (x, y, w, h) so f[2]*f[3] gives the area

    #Pick the face with largest area
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2) #Surround the detected face with a yellow rectangle

        #Crop out the face image (Region of Interst) while keeping an offset margin of 10 pixels
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100,100))

        #Store every 10th image like this
        skip += 1
        if skip%10==0:
            face_data.append(face_section)

    #Show the detected frames and Cropped out frames
    cv2.imshow("Frame", frame)
    cv2.imshow("Face Section", face_section)

    #When the user presses key 's', then terminate this loop
    if skip==50:
        break

#Convert the face list into Numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))

#Save this into the file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Successfully saved data at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows


