#Recognise faces using kNN Algorithm
#Author: Akshay Mattoo

import cv2
import numpy as np
import os

#The kNN Algorithm 
def distance (x1, x2): #Euclidean Distance
    return np.sqrt(((x1-x2)**2).sum())

def kNN (train, test, k=5):
    dist = []

    for i in range (train.shape[0]):
        #Get the vector and its label
        ix = train[i,:-1]
        iy = train[i,-1]

        #Calculate distance from each training point
        d = distance(test, ix)
        dist.append([d, iy])

    #Sort the dist list in increasing order of distances and consider only the k nearest points
    nearest_k = sorted(dist, key=lambda x: x[0])[:k]
    #Retrieve only the labels
    labels = np.array(nearest_k)[:,-1]

    #Find the frequency of each unique label
    prediction = np.unique(labels, return_counts=True)
    #Identify the label with the maximum frequency
    index = np.argmax(prediction[1])
    return prediction[0][index]

#Initialize to capture from the default Web camera
cap = cv2.VideoCapture(0)

#Detect face using Haar cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
path_to_dataset = "...." #Put the path to the location where the generated Numpy arrays (training datasets) are stored.
face_data = []
labels = []

#Maintain the label for each given file
class_ID = 0
#Store a Map from each class_ID to corresponding name
names = {}

#Prepare the dataset
for fx in os.listdir(path_to_dataset):
    if fx.endswith('.npy'): #Identify the Numpy files
        
        #First create the map from class_ID to name
        names[class_ID] = fx[10:-4]
        print ("Loaded "+fx)
        data_item = np.load(path_to_dataset+fx)
        face_data.append(data_item)

        #Create label for the class
        target = class_ID*np.ones((data_item.shape[0],)) #Create a matrix and make each value in it equal to class_ID
        class_ID += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1,1))

train_set = np.concatenate((face_dataset, face_labels), axis=1)

#Testing part
while True:
    ret, frame = cap.read()
    if ret==False:
        continue

    faces = face_cascade.detectMultiScale(frame,1.3, 5)
    if (len(faces)==0):
        continue;

    for face in faces:
        x, y, w, h = face
        
        #Get the region of interest in face
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100,100))

        #Obtain the predicted label
        pred = kNN(train_set, face_section.flatten())

        #Display the prediction (name) of the face and enclose it in a rectangle
        pred_name = names[int(pred)]
        cv2.putText(frame,pred_name,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
