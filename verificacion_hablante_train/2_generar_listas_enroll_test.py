#generate enrollment and testing lists
import numpy as np
import matplotlib.pyplot as plt

#########
#### save a list to txt
def save_list(list2save,filename):
    with open(filename, 'w') as f:
        for item in list2save:
            if type(item[1])==int:
                line = item[0]+'\t'+str(item[1])
                f.write("%s\n" % line)
            else:
                line = item[0]+'\t'+item[1]
                f.write("%s\n" % line)
                
######################################3


path_lists  = 'db_spanish_16k/lists/'
audio2feats = path_lists+'audio2feats.csv'
listFeatsId = path_lists+'FeatsId.csv'

feats2Id = np.loadtxt(listFeatsId,delimiter='\t',dtype=str)

feat_list =feats2Id[:,0]
id_list   =feats2Id[:,1]


spks, count = np.unique(id_list,return_counts=True)
print(f"total speakers: {len(spks)}")

enroll_list   = np.array([])
test_list     = np.array([])
val_list     = np.array([])
#tomar 10 audios de cada hablante para enroll    aleatorios
#tomar 5 audios de cada hablante para test
for ids,spk in enumerate(spks):
#for spk in spks:
    #ids   = int(spk[3:]) #tomar solo el numero
    feats = feat_list[id_list==spk]
    print(spk, len(feats))
    if len(feats)>=5:
        #desordenar
        np.random.shuffle(feats) 
        #si se tiene menos de 10 audios entonces reservar 2 para test y el resto para enroll
        #de lo contrario usar 10 enroll y 5 test
        if feats.size <10:
           L1 = feats[:feats.size-2]
           L2 = feats[feats.size-2:]
           L3 = L1[0]
           L1 = L1[1:]
        else:
           L1 =feats[:10]
           L2 =feats[10:15]
           L3 = feats[15:17]       
        label1 = ids*np.ones(L1.size).astype(int)
        label2 = ids*np.ones(L2.size).astype(int)
        label3 = ids*np.ones(L3.size).astype(int)
        if enroll_list.size==0:
            enroll_list = np.vstack((L1,label1)).T
            test_list   = np.vstack((L2,label2)).T
            val_list   = np.vstack((L3,label3)).T
        else: 
           enroll_list   = np.append(enroll_list,np.vstack((L1,label1)).T,axis=0)
           test_list     = np.append(test_list,np.vstack((L2,label2)).T,axis=0)
           val_list     = np.append(val_list,np.vstack((L3,label3)).T,axis=0)
     

##############

save_list(enroll_list,path_lists+'model.csv')
save_list(test_list,path_lists+'test.csv')
save_list(val_list,path_lists+'val.csv')




