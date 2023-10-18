#prueba para evaluar 
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from glob import glob 
from sklearn.linear_model import LogisticRegression
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
device = 'cpu'


## leer todos los archivos
root = 'Rostro'
file_names = glob(root+"/**/*.png", recursive=True)
clases = glob(root+"/*")
clases.pop(0)
clases = clases=np.array(['-'.join(clase.split('\\')[1].split(' ')[1:3]) for clase in clases])

#leer las imagenes
frames = []
labels = []
dic_clases = {}
for file_name in tqdm(file_names):  
    frame = Image.open(file_name)   
    clase = file_name.split('\\')[1].split(' ')[1:3]
    clase = '-'.join(clase)
    label = np.where(clases==clase)[0][0]
    labels.append(label)
    frames.append(frame)
    dic_clases[label]=clase

print('Done loading')   

#usar  modelos pre-entrenados
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
    )
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)




embeddings = []
names      = []
for i, frame in tqdm(enumerate(frames)):
    x_aligned, prob = mtcnn(frame, return_prob=True)
    if x_aligned is not None:
        #print('Face detected with probability: {:8f}'.format(prob))
     
        x_aligned = torch.stack([x_aligned]).to(device)
        x_embed   = resnet(x_aligned).detach().cpu()
        x_embed   = x_embed.numpy()
        
        #agregar las caracteristicas y la etiqueta a un par de listas 
        embeddings.append(x_embed.ravel())
        names.append(labels[i])

#convertir a array
names = np.array(names)
embeddings = np.array(embeddings)

print(embeddings.shape)
print(names.shape)
ACC = []
for i in range(20):
        #entrenar un clasificador simple
        #usar 80% para entrenar y 20% para validar
        x_train, x_test, y_train, y_test = train_test_split(embeddings,names,test_size = 0.3)

        clf = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf.fit(x_train,y_train)

        y = clf.predict(x_test)
        acc =(y==y_test).sum()/y.size*100
        print('Porcentaje de prediccion : ',acc)
        ACC.append(acc)

ACC=np.array(ACC)
print(ACC.mean(), ACC.std())