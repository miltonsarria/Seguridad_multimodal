#prueba para evaluar si se puede realizar speaker verification 
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from glob import glob 
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
device = 'cpu'


## leer todos los archivos

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
    frame = frame.resize((100,100))   
    frame = np.array(frame).ravel()
    clase = file_name.split('\\')[1].split(' ')[1:3]
    clase = '-'.join(clase)
    label = np.where(clases==clase)[0][0]
    labels.append(label)
    frames.append(frame)
    dic_clases[label]=clase
#convertir a array
names = np.array(labels)
print('Done loading')   
ACC = [] 
X=np.vstack(frames)
print(X.shape)
print(names.shape)
for i in range(20):
    x_train, x_test, y_train, y_test = train_test_split(X,names,test_size = 0.3)
    pca = PCA(n_components=0.99)
    pca.fit(x_train)

    x_train=pca.transform(x_train)
    x_test=pca.transform(x_test)


    #entrenar un clasificador simple
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(x_train,y_train)

    y = clf.predict(x_test)
    acc =(y==y_test).sum()/y.size*100
    print('Porcentaje de prediccion : ',acc)
    ACC.append(acc)
    
ACC=np.array(ACC)
print(ACC.mean(), ACC.std())    