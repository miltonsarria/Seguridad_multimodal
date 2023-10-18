from facenet_pytorch import MTCNN
import torch
import numpy as np
import  cv2
from PIL import Image, ImageDraw
import os
from glob import glob
import torchvision.transforms as T
#from IPython import display

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)
transform = T.ToPILImage()

file_names = glob("Fotos/**/*.*", recursive=True)
clases = glob("Fotos/*")
clases = np.array([clase.split('\\')[1] for clase in clases])
destino = 'todos/'
for clase in clases:
    os.mkdir(destino+clase)

new_dim = (100,100)
continuar = True
k=0;
for i,file_name in enumerate(file_names):
    print(file_name)
    frame = Image.open(file_name)
    if frame.mode=='L':
       print('image in gray')
       frame = frame.convert('RGB')
    boxes, confidence = mtcnn.detect(frame)
    clase = file_name.split('\\')[1]
     
    
    if np.ndim(boxes)!=0:
        for j,box in enumerate(boxes):
            if confidence[j]>0.95:
                print(confidence[j],i)
                f_region  = frame.crop(box)  
                f_region  = f_region.resize(new_dim)
                new_name  = destino+clase +'/img_' + str(k) + '.png'
                k=k+1
                f_region.save(new_name)
       





