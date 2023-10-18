from facenet_pytorch import MTCNN
import torch
import numpy as np
import  cv2
from PIL import Image, ImageDraw
import os


#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)
#habilitar camara
source = 0
cam = cv2.VideoCapture(source)

destino='clientes/';
new_dim = (300,300)
continuar = True
fotos_por_cliente = int(input("Numero de fotos por cliente?: "))
while continuar: 
#for i, frame in enumerate(frames):
    k=0
    cliente = input("Nombre del cliente: ")
    path_cliente = destino + cliente
    if not os.path.exists(path_cliente):
        print("Creandoel directorio para: '{}' ".format(path_cliente))
        os.makedirs(path_cliente)
    while(k<fotos_por_cliente):    
        retval, frame = cam.read()
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Detect faces
        boxes, confidence = mtcnn.detect(frame_pil)
        # Draw faces
        
        #frame = np.array(frame_pil)
        #frame = frame[:, :, ::-1].copy() 
        
        
        
        if np.ndim(boxes)!=0:
            box  = boxes[0]
            c    = confidence[0]
            box = box.astype(int)
            x,y,w,h = box
                
            if x>0 and y>0 and c>0.95:
                    cv2.rectangle(frame, (x,y),(w,h),(0,255,0),2)
                    cv2.imshow('frame',frame)
                    print(f"Probabilidad rostro: {c}" )
                    f_region  = frame_pil.crop(box)  
                    f_region  = f_region.resize(new_dim)
                    new_name  = path_cliente + '/img_' + str(k) + '.png'
                    k=k+1
                    f_region.save(new_name)

                    

        cv2.imshow('frame',frame)
        key =  cv2.waitKey(1)
        if key == 27 :
            break
    cv2.destroyAllWindows()   
    
    cont = input("Desea registrar otro cliente? (S/N): ")
    if cont.upper()=='N':
        continuar = False
       
        
print('\nDone')









