

def pip_auto_install():
    """
    Automatically installs all requirements if pip is installed.
    """
    try:
        from pip._internal import main as pip_main
        pip_main(['install', '-r', 'requirements.txt'])
    except ImportError:
        print("Failed to import pip. Please ensure that pip is installed.")
        print("For further instructions see "
              "https://pip.pypa.io/en/stable/installing/")
        sys.exit(-1)
    except Exception as err:
        print("Failed to install pip requirements: " + err.message)
        sys.exit(-1)


pip_auto_install()


import numpy as np
import os
#from scipy.misc import imread,imresize

from keras.models import model_from_json
import tensorflow as tf
import pickle
import cv2
import sys
dataColor = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX
fx, fy, fh = 10, 50, 45
takingData = 0
className = 'NONE'
count = 0
showMask = 0
# Loading int2word dict
classifier_f = open("int_to_word_out.pickle", "rb")
int_to_word_out = pickle.load(classifier_f)
classifier_f.close()


def load_model():
    
    # load json and create model
    json_file = open('model_face.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_face.h5")
    print("Loaded model from disk")
    return loaded_model

def pre_process(image):
    
    image = image.astype('float32')
    image = image / 255.0
    return image

def load_image():


    img=os.listdir("images")[0]
    image=np.array(cv2.imread("predict/"+img))
    image = cv2.imresize(image, (64, 64))
    image=np.array([image])
    image=pre_process(image)
    return image
def load_video():
    model=load_model()
    cam=cv2.VideoCapture(0)
   
    while 1:
        
        ret2,frame2=cam.read()
        x0, y0, width = 200, 220, 300
        cv2.rectangle(frame2, (x0,y0), (x0+width-1,y0+width-1),dataColor, 12)
     
        roi = frame2[y0:y0+width,x0:x0+width]
        #cv2.imwrite('test.jpg',roi)
        image=roi
        image = cv2.resize(image, (64, 64))
        image=np.array([image])
        image=pre_process(image)
        
        #frame2[y0:y0+width,x0:x0+width] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        pred = model.predict(image)
        pred=int_to_word_out[np.argmax(pred)]
        cv2.putText(frame2, 'Prediction: %s' % (pred), (fx,fy+2*fh), font, 1.0, (245,210,65), 2, 1)
        cv2.imshow('Original', frame2)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
        
        
        
        

    
        
'''
image=load_image()
model=load_model()
prediction=model.predict(image)

print(prediction)
print(np.max(prediction))
print(int_to_word_out[np.argmax(prediction)])
'''
load_video()
