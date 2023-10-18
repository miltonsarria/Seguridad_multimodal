import numpy as np
from scipy import signal
import math
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time 
import pickle
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from facenet_pytorch import MTCNN
warnings.filterwarnings("ignore")

device = 'cpu'
#
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (255,0,0)
thickness              = 2
lineType               = 2


#usar  modelos pre-entrenados

#cargar clasificador entrenado 
fh = open ("clasificador_pca.pkl", "rb")
clf = pickle.load(fh)
fh.close()

fh = open ("modelo_pca.pkl", "rb")
pca = pickle.load(fh)
fh.close()
#etiquetas
fh = open ("clases.pkl", "rb")
clases = pickle.load(fh)
fh.close()

#habilitar camara
source = 0
cam = cv2.VideoCapture(source)
device = 'cpu'
print('Running on device: {}'.format(device))
mtcnn = MTCNN(keep_all=True, device=device)

umbral  = 0.9
new_dim = (100,100)
continuar = True
#realizar un proceso de forma indefinida
while continuar: 
#for i in range(10):
    #
    retval, frame = cam.read()

    frame_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    boxes, confidence = mtcnn.detect(frame_PIL)
    if np.ndim(boxes)!=0:
        #print('Face detected with probability: {:8f}'.format(confidence[0]))
        f_region  = frame_PIL.crop(boxes[0])  
        f_region  = f_region.resize(new_dim)
        X = np.array(f_region).ravel()
        
        X = pca.transform(X.reshape(1, -1))
        
        #se calcula la clase y la probabilidad para saber si es muy baja
        clase     = clf.predict(X.reshape(1, -1))[0]     
        predict   = clf.predict_proba(X)[0]
        texto= 'User: ' + clases[clase] + ' Prob: ' + str(predict[clase])
        #si la probabilidad es muy baja, no asociarlo con ninguno
        #if predict[clase] < umbral:
        #    texto= 'Usuario no autorizado'
        #else:
        #    texto= 'Usuario autorizado: ' + clases[clase]
        
        box = boxes[0].astype(int)
        x,y,w,h = box
            
        if x>0 and y>0 :
          cv2.rectangle(frame, (x,y),(w,h),(0,255,0),2)
          cv2.putText(frame,texto, 
                      bottomLeftCornerOfText, font, 
                      fontScale,fontColor,thickness,
                      lineType)

    cv2.imshow('frame',frame)

    k =  cv2.waitKey(1)
    if k == 27 :
        break
		
print('\nDone')
