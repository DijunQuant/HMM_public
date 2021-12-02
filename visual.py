import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utility import *
from hmmlearn import hmm
import seaborn as sns
import pickle
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from itertools import permutations

flist= frange


featureList=[str((channel,c)) for c in flist for channel in channellist]

coord=pd.read_csv("loc2D.csv",index_col='label')

modelFolder='fittedModels/'
outputFolder='png/'

def getplots(savedModel,savefolder,label,diffonly=False,refmodel=None,makepdf=True):
    freqLabel = {4: '4-7', 8: '8-9', 10: '10-13', 14: '14-17', 18: '18-24', 25: '25-29', 30: '30-39', 40: '40-50'}
    meanArray=getMeanArray(savedModel)
    n=len(meanArray)
    if refmodel!=None:
        refmeans = getMeanArray(refmodel)
        order, diff = sortMeans(refmeans, meanArray)
        print('diff to refmodel: ',diff)
        meanArray=meanArray[order, :, :]
        if diffonly:
            return
    with PdfPages(savefolder+label+'.pdf') as export_pdf:
        for state in range(n):
            fig, ax = plt.subplots(3, 3, figsize=(15, 15))

            #plt.suptitle(label+'_'+str(state))
            meanPD = pd.DataFrame(meanArray[state],index=channellist, columns=flist)
            meanPD.loc[:, 'x'] = coord.loc[meanPD.index, 'x'].values
            meanPD.loc[:, 'y'] = coord.loc[meanPD.index, 'y'].values
            x_index = 0
            y_index = 0
            for freq in flist:
                print(x_index,y_index)
                CS = plotContour(meanPD['x'].values, meanPD['y'].values, meanPD[freq].values, ax[x_index, y_index], fig,
                                 makebar=False)
                ax[x_index, y_index].set_title(freqLabel[freq],fontdict={'fontsize':20, 'fontweight': 'bold'})
                if (x_index==1) & (y_index==1):
                    x_index = 2
                    y_index = 0
                else:
                    y_index += 1
                    if (y_index == 3):
                        x_index += 1
                        y_index = 0
            #ax[1, 2].figure.colorbar(CS, pad=.05, extend='both', fraction=0.5, ticks=np.linspace(-1, 1, 11))
            ax[1, 2].axis('off')
            fig.colorbar(CS, ax=ax[1, 2], pad=.05, extend='both', fraction=0.5, ticks=np.linspace(-1, 1, 9))
            plt.tight_layout()
            #plt.savefig(folder + 'png/' + str(state + 1) + '_6' + '.png')
            if makepdf:
                export_pdf.savefig()
            else:
                plt.savefig(savefolder + label +'_'+ str(state + 1) +'.png')
            plt.close()
def getMeanArray(model):
    n=len(model.means_)
    allmeans=[]
    for state in range(n):
        meanSer=pd.Series(model.means_[state],index=featureList)
        meanPD=pd.DataFrame([[meanSer.loc[str((channel,c))] for c in flist] for channel in channellist],
                   index=channellist,columns=flist)
        allmeans.append(meanPD.values)
    return np.array(allmeans)
def sortMeans(benchmarkMean,modelMeans):
    n=len(benchmarkMean)
    n_orders=list(permutations(range(n)))
    difflist=[np.linalg.norm(benchmarkMean-modelMeans[l,:,:],axis=(1,2)).mean() for l in n_orders]
    minindex=np.argmin(difflist)
    mindiff=difflist[minindex]
    return n_orders[minindex],mindiff


#######
##  make topographies and save to pdf for a fitted model
#######

label='7_all_init_9_iter_50'
with open(modelFolder+label+".pkl", "rb") as file:
    savedModel=pickle.load(file)
getplots(savedModel,outputFolder,label,diffonly=False,refmodel=None,makepdf=True)

#######
##  make topographies and save to pdf for a fitted model, order the states to match the refmodel
#######


label='7_all_init_80_iter_50_b'
with open(modelFolder+label+".pkl", "rb") as file:
    model=pickle.load(file)
getplots(model,outputFolder,label,diffonly=False,refmodel=savedModel)
