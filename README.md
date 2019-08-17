# Face_recognition

The Python script "gen_train_data.py" reads and shows video stream (from the default web camera). All faces from the frame are detected using Haar feature-based cascade classifiers (for more information refer https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html). 

The faces are bounded by a rectangle. The largest face image is flattened, and saved as a Numpy array. 

The above procedure can be repeated multiple times to generate the training data

The script "face_recog.py" takes the above generated data, creates class ID and identifies name. Then using the Web camera, it implements kNN algorithm to match the face with a training data. This identified face is enclosed in a rectangle and the name is shown on top.
