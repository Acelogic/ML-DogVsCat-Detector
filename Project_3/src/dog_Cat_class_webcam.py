#Import Relevant Libraries
import numpy as np
import cv2
import time

#loading our saved model from notebooks
from tensorflow.keras.models import load_model
classifier = load_model('dogcat_model_bak.h5')



def main():
    confidence_threshold = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_time = time.time()
    frame_count = 0

    font = cv2.FONT_HERSHEY_SIMPLEX

    '''To test the model on the webcam'''
    video_capture = cv2.VideoCapture(1)

    while True:
        '''Test: WEBCAM '''
        _, imgo= video_capture.read()
        
        img11 = cv2.resize(imgo, (800,800))
        
        img = cv2.resize(imgo, (128,128))

        #get image shape
        frame_count +=1
        
        img = np.array(img)
        img = img/255
        
        # create a batch of size 1 [N,H,W,C]
        img = np.expand_dims(img, axis=0)
        prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
        
        print(prediction)
        
        if(prediction[:,1]>confidence_threshold):
            value ='Dog :%1.2f'%(prediction[0,0])
            
            cv2.putText(img11, 
                value, 
                (35, 35), 
                font,1.5, 
                (255, 255, 0), 
                2, 
                cv2.LINE_AA)
        elif(prediction[:,0]>confidence_threshold):
            value ='Cat :%1.2f'%(1.0-prediction[0,0])
            cv2.putText(img11, 
                value, 
                (35, 35), 
                font, 1.5, 
                (0, 255, 255), 
                2, 
                cv2.LINE_AA)
        else:
            value ='None'
            cv2.putText(img11, 
                value, 
                (35, 35), 
                font, 1.5, 
                (0, 255, 255), 
                2, 
                cv2.LINE_AA)
            
        elapsed_time = time.time() - start_time
        fps = frame_count/elapsed_time
        print ("fps: ", str(round(fps, 2)))
        cv2.imshow("Image", img11)
        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break
    cv2.destroyAllWindows()

main()
