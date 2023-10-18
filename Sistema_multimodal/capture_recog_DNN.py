import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from PIL import Image
import time 
import pickle
import warnings
import pyttsx3 
from tools import getAudio
import pyttsx3
import pyaudio
import h5py  as h5
warnings.filterwarnings("ignore")

## iniciar motor de tts
engine_tts = pyttsx3.init()
voices = engine_tts.getProperty('voices')
engine_tts.setProperty('voice', voices[2].id)
engine_tts.setProperty('rate', 130)


device = 'cpu'
path_modelos_rostros = "modelos/rostros/"
path_modelos_voz     = "modelos/hablantes/"
feats                =  'xvec_1'
ref_model            = path_modelos_voz+"reference_models_"+feats+".h5"
#objeto de pyAudio
p             = pyaudio.PyAudio()
fs            = 16000 #Hertz
duracion      = 4#cuantos segundos por audio?

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

#simlilaridad coseno
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
classifier_spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")


fh = open (path_modelos_rostros+"clasificador_DNN.pkl", "rb")
clf = pickle.load(fh)
fh.close()

#etiquetas
fh = open (path_modelos_rostros+"clases.pkl", "rb")
clases = pickle.load(fh)
fh.close()

#habilitar camara
source = 0
cam = cv2.VideoCapture(source)

umbral  = 0.8

continuar = True
#realizar un proceso de forma indefinida
text = "Bienvenido a nuestro sistema, por favor ubicate frente a la camara para identificarte"
engine_tts.say(text)
engine_tts.runAndWait()
verification = False
while continuar: 

    retval, frame = cam.read()
    frame_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    
    #use new models
    boxes, confidence = mtcnn.detect(frame_PIL)
    x_aligned, prob = mtcnn(frame_PIL, return_prob=True)
    
    if x_aligned is not None:
       
        x_aligned = torch.stack([x_aligned]).to(device)
        x_embed   = resnet(x_aligned).detach().cpu() 
        
        #se calcula la clase y la probabilidad para saber si es muy baja
        clase     = clf.predict(x_embed)[0]     
        predict   = clf.predict_proba(x_embed)[0]
        #texto= 'USER: ' + clases[clase] + ' Prob: ' + str(predict[clase])
        #si la probabilidad es muy baja, no asociarlo con ninguno
        if predict[clase] < umbral:
            texto= 'Usuario no autorizado'
        else:
            texto= 'Usuario autorizado: ' + clases[clase]
            text = "Hola " + clases[clase] + ", hablar en el microfono durante 4 segundos"
            #inicia verificacion hablante
            engine_tts.say(text)
            engine_tts.runAndWait()
            x=getAudio(p,RATE=fs,RECORD_SECONDS=duracion)
            signal     = torch.from_numpy(x)
            speakerTest= classifier_spk.encode_batch(signal)
            speakerTest=speakerTest[0,0,:]
            with h5.File(ref_model, "r") as f:            
                 data = np.array(f[clases[clase]])
                 th   = np.array(f["th"])
            th          = torch.from_numpy(th)
            enroll_data = torch.from_numpy(data)
            scores = cos(enroll_data, speakerTest)        
            verification = scores > th
            if verification:
                text = "Verificado con exito"
                engine_tts.say(text)
                engine_tts.runAndWait()
            #finaliza verificacion hablante
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
    if verification:
        cv2.destroyAllWindows()	
print('\nDone')