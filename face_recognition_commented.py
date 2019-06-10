# Face Recognition

import cv2

#what is a cascade -- it is basically a series of filter that is required for detection.
#so we make objects of cascade (both eye and for face recognition).
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # We load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # We load the cascade for the eyes.
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')# We load the cascade for the smile.

#functions to detect the required attribute .
#cascade works on black and white image.
def detect(gray, frame): # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles.
    
    #The detectMultiScale method will return the co-ordinates of the rectangle detecting a face.
    #The second parameter in the method represents the scale factor i.e. by how much the size of image is going to be reduced or equivalently by how much will the size of filter be increased.
    #The last Parameter denotes the neighbour , the box will only get accepted only if its 'n' numbers of neighbours are accepted.
    #The values for these parameters are experimental or by hit and trial.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    
    for (x, y, w, h) in faces: # For each detected face:
        
        #we have a built in function in openCV for the creation of rectangle.
        #1st arg - on which image do you want to draw the rectangle.
        #2nd arg - the upper left co-ordinate of the rectangle.
        #3rd arg - the lower right co-ordinate of the rectangle.
        #4th arg - color for the rectangle.
        #5th arg - thickness of edges of rectangle.
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
        
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # We get the region of interest in the colored image.
        
        #Here we are detecting eye and smile in the reference of our faces and not the whole frame.
        #Instead of face cascade we are having eye cascade and smile cascade.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22) # We apply the detectMultiScale method to locate one or several eyes in the image.
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22) # We apply the detectMultiScale method to locate smile in the image.
        
        #now we aill draw the rectangles for the eyes.
        for (ex, ey, ew, eh) in eyes: # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) # We paint a rectangle around the eyes, but inside the referential of the face.
        
        #now we aill draw the rectangles for the smile.
        for (ex, ey, ew, eh) in smiles:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 0, 255), 2)# We paint a rectangle around the smile, but inside the referential of the face.
            
    return frame # We return the image with the detector rectangles.

#The parameter here accepts 2 values i.e. 0--if internal web cam is used 1-- if external web cam is  used .
video_capture = cv2.VideoCapture(0) # We turn the webcam on.

while True: # We repeat infinitely (until break):
    
    #the video_capture.read() function returns two values, the second value is the last frame returned and since only that is required so we ignore the first value.
    _, frame = video_capture.read() # We get the last frame.
    
    #The second parameter of the cvtColor function takes the average of the colors to produce right shades of gray.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
    
    canvas = detect(gray, frame) # We get the output of our detect function.
    
    cv2.imshow('Video', canvas) # We display the outputs.
    
    #in the wait function the smallest value that we can give is 1 i.e. 1 millisec if we put 0 or -1 will block until you press a key.
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.

video_capture.release() # We turn the webcam off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.