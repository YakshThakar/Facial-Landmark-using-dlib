import numpy as np  
import cv2  
import dlib  

image_path = cv2.VideoCapture(0)  
cascade_path = "haarcascade_frontalface_alt2.xml"  
predictor_path= "shape_predictor_68_face_landmarks.dat"  

# Create the haar cascade  
faceCascade = cv2.CascadeClassifier(cascade_path)  

# create the landmark predictor  
predictor = dlib.shape_predictor(predictor_path)  

  
while True:
    ret,image=image_path.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5,minSize=(100, 100),flags=cv2.CASCADE_SCALE_IMAGE)
    print("Face Found {0} faces!".format(len(faces)))
# Draw a rectangle around the faces  
    for (x, y, w, h) in faces:
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))   
        detected_landmarks = predictor(image, dlib_rect).parts()  
        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])  
        for idx, point in enumerate(landmarks):  
            pos = (point[0, 0], point[0, 1])
 
   # draw points on the landmark positions  
            cv2.circle(image, pos, 2, color=(255, 0, 255))  
 
    
    cv2.imshow("Landmarks", image)  
    cv2.waitKey(1) 

image_path.release()
cv2.destroyAllWindows()
