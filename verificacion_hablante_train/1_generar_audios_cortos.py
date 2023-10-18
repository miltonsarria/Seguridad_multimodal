import os
from glob import glob
import shutil
from  librosa import load, resample
from soundfile import write
import numpy as np
import matplotlib.pyplot as plt
 ######################
 #### save a list to txt
def save_list(list2save,filename):
    with open(filename, 'a') as f:
        for item in list2save:
            if type(item[1])==int:
                line = item[0]+'\t'+str(item[1])
                f.write("%s\n" % line)
            else:
                line = item[0]+'\t'+item[1]
                f.write("%s\n" % line)
 #### generar audios de x segundos por cada registro de un hablante
 
source_path = 'BD/'
audioList   = glob(source_path  +"**/*.wav",recursive=True)
clases      = np.loadtxt(source_path+"listaBD.csv",delimiter=',',dtype=str)
path_to_id=dict()
for item in clases:
    path_to_id[item[0]]=item[1]

destination  = 'db_spanish_16k/'  
audio2feats=[]
listFeatsId=[]
##########################
duracion = 8  # segundos
#########################

for entry in audioList:
    #load signal
    x, fs = load(entry,sr=None)
    x=resample(x, orig_sr=fs, target_sr=16000)
    fs=16000
    window_len = int(duracion*fs) #ventana de x segundos
    ## obtener el nombre del archivo y el id del hablante
    base_name = os.path.basename(entry)
       
    path_spk = entry.split('\\')[1]
    #print(entry,base_name,path_spk) 
    id_spk =  path_to_id[path_spk]
    #define destination path for chunks
    dest_path_spk = destination + id_spk +'/'
    
    if not os.path.exists(dest_path_spk):
  
        print("creating '{}' directory".format(dest_path_spk),end="\r")
        os.makedirs(dest_path_spk)
    
    ini   = 0
    fin   = window_len
    count = len(glob(dest_path_spk+"**/*.wav",recursive=True))
    if (x.size<window_len):
        #si es muy corto igual se agrega
        file_name = dest_path_spk + id_spk + '_'+ str(count) + '.wav'
        write(file_name,x,fs) 
        feat_file = id_spk + '_'+ str(count) 
        audio2feats.append([file_name,feat_file])
        listFeatsId.append([feat_file,id_spk])
        
    while (fin<x.size):
        chunk = x[ini:fin]
                                    #+ 
        file_name = dest_path_spk + id_spk + '_'+ str(count) + '.wav'
        write(file_name,chunk,fs)
        
        feat_file = id_spk + '_'+ str(count) 
        #generate lists
        audio2feats.append([file_name,feat_file])
        listFeatsId.append([feat_file,id_spk])
        
        #advance
        ini = fin
        fin = fin +window_len
        count = count + 1
    
    
     
############
path_lists = destination+'lists/'
if not os.path.exists(path_lists):
        print("creating '{}' directory".format(path_lists))
        os.makedirs(path_lists)
save_list(audio2feats,path_lists+'audio2feats.csv') 
save_list(listFeatsId,path_lists+'FeatsId.csv')      
     
