import numpy as np
from scipy import signal
import math
import matplotlib.pyplot as plt
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from PIL import Image
import time 
import pickle
import warnings
warnings.filterwarnings("ignore")

device = 'cpu'

#usar  modelos pre-entrenados
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
    )
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
#cargar clasificador entrenado
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (255,0,0)
thickness              = 2
lineType               = 2



fh = open ("clasificador_DNN.pkl", "rb")
clf = pickle.load(fh)
fh.close()

#etiquetas
fh = open ("clases.pkl", "rb")
clases = pickle.load(fh)
fh.close()

#habilitar camara
source = 0
cam = cv2.VideoCapture(source)

umbral  = 0.8

continuar = True
#realizar un proceso de forma indefinida
while continuar: 

    retval, frame = cam.read()
    frame_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    
    #use new models
    boxes, confidence = mtcnn.detect(frame_PIL)
    x_aligned, prob = mtcnn(frame_PIL, return_prob=True)
    
    if x_aligned is not None:
        #print('Face detected with probability: {:8f}'.format(prob))
        
        x_aligned = torch.stack([x_aligned]).to(device)
        x_embed   = resnet(x_aligned).detach().cpu() 
        
        #se calcula la clase y la probabilidad para saber si es muy baja
        clase     = clf.predict(x_embed)[0]     
        predict   = clf.predict_proba(x_embed)[0]
        texto= 'USER: ' + clases[clase] + ' Prob: ' + str(predict[clase])
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