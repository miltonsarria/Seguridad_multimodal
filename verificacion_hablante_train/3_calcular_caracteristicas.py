import torchaudio
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import EncoderClassifier
import h5py  as h5
import numpy as np
from tqdm import tqdm
import os

#verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
#usar dos modelos pre-entrenados
#classifier0 = EncoderClassifier.from_hparams(source="model", 
#                                             hparams_file='hparams_inference.yaml', 
#                                              savedir="model")
                                            
classifier1 = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
classifier2 = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")


#encodings para facemask
audio_path=''
feats_path='features/'
if not os.path.exists(feats_path):
        print("creating '{}' directory".format(feats_path))
        os.makedirs(feats_path)
        
#read list of audio files to encode
list_waves_file='lists/audio2feats.csv'
fh = open(list_waves_file,'r')
lines = fh.readlines()
fh.close()

for line in tqdm(lines):
#for i in range(1):
    audio_file, feat_file = line.split('\t')
    
    audio_file = audio_path + audio_file
    feat_file  = feats_path + feat_file[:-1] + '.h5'
    #get signal and encode
    #print('encoding: ',audio_file, '  to file: ',feat_file)
    signal, fs =torchaudio.load(audio_file)
    #x0_v = classifier0.encode_batch(signal)
    x1_v = classifier1.encode_batch(signal)
    x2_v = classifier2.encode_batch(signal)
    
    #x0_v = np.array(x0_v[0,0,:])
    x1_v = np.array(x1_v[0,0,:])
    x2_v = np.array(x2_v[0,0,:])
    
    hf   = h5.File(feat_file, "a") #handle file
    try:
        #dset = hf.create_dataset('xvec_0', data=x0_v) 
        dset = hf.create_dataset('xvec_1', data=x1_v)
        dset = hf.create_dataset('xvec_2', data=x2_v)
    except:
        print('ya existe')
    hf.close()

    del x1_v,x2_v, hf, dset


