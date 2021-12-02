import pandas as pd
import numpy as np
import os
from utility import *
from hmmlearn import hmm
import pickle
import sys


includeTypes=['TO','ANA','INS']
hmm_n=7 #number of hidden state
hmm_niter=20 #maximum number of iterations
randomseed=0

folder='PATHTOINPUTDATA/nFeature/'
modeloutput='PATHTOOUTPUTDATA/fittedModels/'

flist=[4,8,10,14,18,25,30,40] #include all frequency bands

featureList=[str((channel,c)) for c in flist for channel in channellist]

featureDf=pd.DataFrame()
nlist=[]

for f in os.listdir(folder):
    if f.split('.')[1]!='csv':continue
    if not (f.split('.')[0].split('_')[1] in includeTypes):continue
    tmp=pd.read_csv(folder+f,index_col=None,header=[0,1])
    #tmp = pd.read_csv(this_folder + f, index_col=None)
    tmp.columns=[("(\'"+col[0]+"\', "+str(col[1])+")" if len(col[1])<3 else col[0]) for col in tmp.columns.values]
    lenlist = tmp['n'].value_counts()
    featureDf=pd.concat([featureDf,tmp[featureList]])
    nlist=nlist+list(lenlist[tmp['n'].unique()].values)

#print(featureDf.info())

datacov=featureDf[featureList].cov().values

model = hmm.GaussianHMM(n_components=hmm_n, n_iter=hmm_niter, covariance_type='tied', params="stm",init_params='stm',random_state=randomseed)
model.covars_ = datacov
model.fit(featureDf[featureList].values, nlist)
with open(modeloutput + str(hmm_n)+'_init_'+str(randomseed)+'_iter_'+str(hmm_niter)+'.pkl', "wb") as file:
        pickle.dump(model, file)
