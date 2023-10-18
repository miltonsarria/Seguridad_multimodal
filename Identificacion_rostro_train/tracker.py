from facenet_pytorch import MTCNN
import torch
import numpy as np
import  cv2
from PIL import Image, ImageDraw
#from IPython import display

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)
#habilitar camara
source = 0
cam = cv2.VideoCapture(source)



continuar = True
while continuar: 
#for i, frame in enumerate(frames):
    retval, frame = cam.read()
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Detect faces
    boxes, confidence = mtcnn.detect(frame_pil)
    # Draw faces
    if np.ndim(boxes)!=0:
    
        for box,c in zip(boxes,confidence):
           
            box = box.astype(int)
            x,y,w,h = box
            
            if x>0 and y>0 and c>0.90:
                cv2.rectangle(frame, (x,y),(w,h),(0,255,0),2)
            

    cv2.imshow('frame',frame)

    k =  cv2.waitKey(1)
    if k == 27 :
        break
		
print('\nDone')

cv2.destroyAllWindows()







