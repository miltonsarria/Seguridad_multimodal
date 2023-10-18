from tools import *
import torchaudio
import h5py  as h5
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve
from sklearn.metrics import DetCurveDisplay
from speechbrain.utils.metric_stats import EER

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

root_feats   ='features/'
#source_path = 'hablantes2/'
#lists
path_lists   = 'lists/'
clients_list = path_lists+'model.csv';
test_list    = path_lists+'/test.csv';

######
field = 'xvec_1'
###########
print('[INFO] load data')
#carga clientes
feats_list, id_spk_clients = readList(clients_list)
data_clients = loadFeats(root_feats,field,feats_list)
#carga prueba
feats_list, id_spk_test = readList(test_list)
data_test = loadFeats(root_feats,field,feats_list)

#clases      = np.loadtxt(source_path+"lista_aux.csv",delimiter=',',dtype=str)
#id_to_name=dict()
#for item in clases:
#    ids   = int(item[1][3:])
#    id_to_name[ids]=item[0]
    
print(f'[INFO] done, dims = {data_clients.shape[1]}')

#average model xvectors
enrol_persons = np.unique(id_spk_clients)
en_N = enrol_persons.size
enrol_data_avr = np.zeros((en_N, data_clients.shape[1]))
print(f'[INFO] average clients and save reference models')
ref_model = "reference_models_"+field+".h5"
hf   = h5.File(ref_model, "a") #handle file
for i,id_sp in enumerate(enrol_persons):
    indx = id_sp == id_spk_clients
    spk_data = data_clients[indx,:]
    enrol_data_avr[i,:] = spk_data.mean(axis=0)
    '''
    print(id_to_name[id_sp])
    
    try:
      dset = hf.create_dataset(id_to_name[id_sp], data=spk_data.mean(axis=0)) 
    except:
        print('ya existe')
    '''

print(f'[INFO] done!')    
#enrol_data_avr = torch.from_numpy(enrol_data_avr)
#data_test      = torch.from_numpy(data_test)

eer,min_dcf, y_scores, y_true, total_trials,th = cosineValidation(enrol_data_avr,data_test,enrol_persons,id_spk_test)

print(f"cosine EER = {eer*100}\nTotal pruebas={total_trials}\n\nmin_dcf = {min_dcf}\nth={th}")

try:
      dset = hf.create_dataset("th", data=th) 
except:
      print('ya existe')
hf.close()

DetCurveDisplay.from_predictions(y_true, y_scores)
plt.show()

nbins = 50

h1 = plt.hist(y_scores[y_true==1],nbins,density=True,color="blue")
h0 = plt.hist(y_scores[y_true==0],nbins,density=True,color="green")

sigma1 = np.std(y_scores[y_true==1])
mu1    = np.mean(y_scores[y_true==1])
y1 = ((1 / (np.sqrt(2 * np.pi) * sigma1)) *
     np.exp(-0.5 * (1 / sigma1 * (h1[1] - mu1))**2))

plt.plot(h1[1],y1)

sigma0 = np.std(y_scores[y_true==0])
mu0    = np.mean(y_scores[y_true==0])
y0 = ((1 / (np.sqrt(2 * np.pi) * sigma0)) *
     np.exp(-0.5 * (1 / sigma0 * (h0[1] - mu0))**2))

plt.plot(h0[1],y0)     
     
plt.show()

