import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.interpolate

import matplotlib

channellist=['FP1','FP2','F7','F3','FZ','F4','F8','T7','C3','CZ','C4','T8','P7','P3','PZ','P4','P8','O1','O2']

#frange=[1,2,3,4,5,6,8,11,16,23,32] #band frequency
#frange=[3,4,5,8,10,13,20,30,41]
#frange=[3,4,5,8,11,16,23,32,41]
frange=[4,8,10,14,18,25,30,40]


frangeall=frange
prestim={'INS':1,'ANA':1,'TO':1}
#presetimulus = 2 for solved, 1 for unsolved

def checkExclude(subid, f):
    #if subid=='126':
    #   if f=='TO_88':return True
    return False

def preprocess(df,dswindow,frange,prestimulus,mawindow, difflist=[1,2,3,4,5],plot=False):
    df['toOnset']=df[0]-df[0].values[0]-prestimulus
    df.rename(columns={0:'toRsp'},inplace=True)
    df.set_index('toOnset',inplace=True)
    tmp=df[['toRsp']]
    for k in mawindow.keys():
        df[k] = df[k].rolling(mawindow[k]).mean().fillna(method='bfill')
    for i in range(len(frange) - 1):
        tmp[frange[i]] = df[list(range(frange[i], frange[i + 1]))].mean(axis=1)
    tmp[frange[-1]] = df[list(range(frange[-1], 50))].mean(axis=1)
    tmp=tmp.iloc[dswindow-1::dswindow]
    for f in difflist:#use negative frequency for diff
        tmp[-f]=tmp[f].diff()
    if plot:
        plt.subplots(1,2,figsize=(20,6))
        ax1=plt.subplot(1,2,1)
        tmp[frange[:4]].plot(ax=ax1).legend(bbox_to_anchor=(1, 1))
        ax2=plt.subplot(1,2,2)
        tmp[frange[:-3]].plot(ax=ax2).legend(bbox_to_anchor=(1, 1))
    return tmp

def plotContour(x,y,z,ax,fig,makebar):
    # some parameters
    N = 300             # number of points for interpolation
    xy_center = [0,0]   # center of the plot
    radius = 1          # radius
    xi = np.linspace(-1, 1, N)
    yi = np.linspace(-1, 1, N)
    zi = scipy.interpolate.griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    # set points > radius to not-a-number. They will not be plotted.
    # the dr/2 makes the edges a bit smoother
    dr = xi[1] - xi[0]
    for i in range(N):
        for j in range(N):
            r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
            if (r - dr/2) > radius:
                zi[j,i] = "nan"
    # make figure
    #fig = plt.figure()
    # set aspect = 1 to make it a circle
    #ax = fig.add_subplot(111, aspect = 1)
    # use different number of levels for the fill and the lines

    CS = ax.contourf(xi, yi, zi, levels=np.linspace(-1,1,100), cmap=plt.cm.jet, zorder=1, vmin=-1, vmax=1,extend='both')
    ax.contour(xi, yi, zi, 15, colors = "grey", zorder = 2)
    # make a color bar
    if makebar:
        cbar = fig.colorbar(CS, ax=ax,ticks=np.linspace(-1.5,1.5,11))
    # add the data points
    # I guess there are no data points outside the head...
    ax.scatter(x, y, marker = 'o', c = 'b', s = 15, zorder = 3)

    # draw a circle
    # change the linewidth to hide the
    circle = matplotlib.patches.Circle(xy = xy_center, radius = radius, edgecolor = "k", facecolor = "none")
    ax.add_patch(circle)

    # make the axis invisible
    for loc, spine in ax.spines.items():
        # use ax.spines.items() in Python 3
        spine.set_linewidth(0)
    # remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add some body parts. Hide unwanted parts by setting the zorder low
    # add two ears
    circle = matplotlib.patches.Ellipse(xy = [-1,0], width = 0.25, height = .5, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    circle = matplotlib.patches.Ellipse(xy = [1,0], width = 0.25, height = .5, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    # add a nose
    xy = [[-.25,.75], [0,1.15],[.25,.75]]
    polygon = matplotlib.patches.Polygon(xy = xy, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(polygon)

    # set axes limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    return CS
    #plt.tight_layout()
    #plt.show()

def getContour(x,y,z,fig,index):
    # some parameters
    N = 300             # number of points for interpolation
    xy_center = [0,0]   # center of the plot
    radius = 1          # radius
    xi = np.linspace(-2, 2, N)
    yi = np.linspace(-2, 2, N)
    zi = scipy.interpolate.griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    # set points > radius to not-a-number. They will not be plotted.
    # the dr/2 makes the edges a bit smoother
    dr = xi[1] - xi[0]
    for i in range(N):
        for j in range(N):
            r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
            if (r - dr/2) > radius:
                zi[j,i] = "nan"
    # set aspect = 1 to make it a circle
    ax = fig.add_subplot(index, aspect = 1)
    # use different number of levels for the fill and the lines
    CS = ax.contourf(xi, yi, zi, 60, cmap = plt.cm.jet, zorder = 1)
    ax.contour(xi, yi, zi, 15, colors = "grey", zorder = 2)
    # make a color bar
    #cbar = fig.colorbar(CS, ax=ax)

    # add the data points
    # I guess there are no data points outside the head...
    ax.scatter(x, y, marker = 'o', c = 'b', s = 15, zorder = 3)
    # draw a circle
    # change the linewidth to hide the
    circle = matplotlib.patches.Circle(xy = xy_center, radius = radius, edgecolor = "k", facecolor = "none")
    ax.add_patch(circle)
    # make the axis invisible
    for loc, spine in ax.spines.items():
        # use ax.spines.items() in Python 3
        spine.set_linewidth(0)
    # remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # Add some body parts. Hide unwanted parts by setting the zorder low
    # add two ears
    circle = matplotlib.patches.Ellipse(xy = [-1,0], width = 0.25, height = .5, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    circle = matplotlib.patches.Ellipse(xy = [1,0], width = 0.25, height = .5, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    # add a nose
    xy = [[-.25,.75], [0,1.25],[.25,.75]]
    polygon = matplotlib.patches.Polygon(xy = xy, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(polygon)
    # set axes limits
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    return ax
def computeNorm(subid,solvetypes,dswindow,sourcefolder,outputfolder,mawindow={}):
    for f in os.listdir(sourcefolder +subid + '/'):
        solvetype=f.split('_')[0]
        n=f.split('_')[1]
        if solvetype in solvetypes:
            fname = sourcefolder + subid + '/' + f
            if checkExclude(subid,f):continue
            #if os.path.exists(folder + 'var_' + subid + '/' + solvetype + '_' + str(n) + '.csv'):continue
            if not os.path.exists(outputfolder + 'var_' + subid):
                os.makedirs(outputfolder + 'var_' + subid)
                os.makedirs(outputfolder + 'mean_' + subid)
            fulldict = dict()
            #if True:
            try:
                for channel in channellist:
                    # print(channel)
                    tmp = preprocess(
                        pd.read_csv(fname + '/' + subid + '_' + solvetype + '_' + str(n) + '_' + channel + '.csv',
                                    header=None).T, \
                        dswindow, frange,prestim[solvetype],mawindow,difflist=[])
                    fulldict[channel] = tmp
                covDF = pd.DataFrame()
                meanDF = pd.DataFrame()
                for f in frangeall:
                    tmp = pd.DataFrame({k: fulldict[k][f] for k in fulldict.keys()})
                    covtmp = tmp.cov()
                    covtmp['freq'] = f
                    covDF = pd.concat([covDF, covtmp])
                    meantmp = tmp.mean()
                    meanDF = pd.concat([meanDF, pd.DataFrame(meantmp).T])
                meanDF['freq'] = frangeall
                meanDF.set_index('freq').to_csv(outputfolder + 'mean_' + subid + '/' + solvetype + '_' + str(n) + '.csv')
                covDF.to_csv(outputfolder + 'var_' + subid + '/' + solvetype + '_' + str(n) + '.csv')
            except Exception as e:
                print(fname, channel)
                print(e)

def calBIC(ll,n,nfeature,nSample):
    vk= (nfeature + nfeature ** 2 + n + 1)*n #number of parameters
    return ll-1/2*vk*np.log(nSample)


def avgNorm(subid,folder):
    #avg var across trials
    diagdf_ana = pd.DataFrame()
    diagdf_ins = pd.DataFrame()
    diagdf_to = pd.DataFrame()
    for f in os.listdir(folder + '/var_' + subid + '/'):
        if f.split('.')[1]!='csv':continue
        df = pd.read_csv(folder + '/var_' + subid + '/' + f, index_col=0)
        tmp = pd.DataFrame([[df[df['freq'] == freq].loc[f, f] for f in channellist] for freq in frangeall],
                           columns=channellist)
        tmp['n'] = f.split('.')[0].split('_')[1]
        tmp['freq'] = frangeall
        tmp['trial']=f.split('.')[0]
        if f.split('_')[0] == 'ANA':
            diagdf_ana = pd.concat([diagdf_ana, tmp])
        else:
            if f.split('_')[0] == 'INS':
                diagdf_ins = pd.concat([diagdf_ins, tmp])
            else:
                diagdf_to = pd.concat([diagdf_to, tmp])
    diagdf_all = pd.concat([diagdf_ana, diagdf_ins, diagdf_to])
    #filter outlier
    m = diagdf_all.groupby('freq').mean()
    sd = diagdf_all.groupby('freq').std()
    badrows = []
    for i, r in diagdf_all.iterrows():
        freq = r['freq']
        if (r[channellist] - (m.loc[freq, channellist] + 8 * sd.loc[freq, channellist])).max() > 0:
            badrows.append([r['trial'], freq,
                            channellist[np.argmax((r[channellist] -
                                                   (m.loc[freq, channellist] + 8 * sd.loc[freq, channellist])).values)]])
    if len(badrows)>0:
        badrowsDF=pd.DataFrame(badrows,columns=['trial','freq','channel'])
        badrows=badrowsDF['trial'].unique()
    diagdf_all=diagdf_all[~diagdf_all['trial'].isin(badrows)]
    # norm per freq per channel
    pd.DataFrame([np.sqrt(diagdf_all[diagdf_all['freq'] == f][channellist].mean()) for f in frangeall],
                 index=frangeall).to_csv(folder + 'norm/' + subid + '_sd.csv')
    #avg mean across trials
    diagdf_ana = pd.DataFrame()
    diagdf_ins = pd.DataFrame()
    diagdf_to = pd.DataFrame()
    for f in os.listdir(folder + '/mean_' + subid + '/'):
        if (f.split('.')[0] in badrows): continue
        df = pd.read_csv(folder + '/mean_' + subid + '/' + f, index_col=0)
        if f.split('_')[0] == 'ANA':
            diagdf_ana = pd.concat([diagdf_ana, df])
        else:
            if f.split('_')[0] == 'INS':
                diagdf_ins = pd.concat([diagdf_ins, df])
            else:
                diagdf_to = pd.concat([diagdf_to, df])
    diagdf_all = pd.concat([diagdf_ana, diagdf_ins,diagdf_to])
    # norm per freq per channel
    pd.DataFrame([diagdf_all.loc[f, channellist].mean() for f in frangeall], index=frangeall).to_csv(
        folder + 'norm/' + subid + '_m.csv')
    if len(badrows)>0: badrowsDF.to_csv(folder+'bad/'+subid+'.csv')

#stepsize 40ms
def computeStats(df,stepsize=40):
    zscore=pd.DataFrame()
    duration=pd.DataFrame()
    duration_se=pd.DataFrame()
    ndf={}
    std={}
    n=len(df['state'].unique())
    for style in ['INS','ANA','TO']:
        tmp=df[df['style']==style]
        statecnt=tmp.groupby('state').count()[1]
        duration.loc[:,style]=(statecnt/statecnt.sum())
        ndf[style]=len(tmp['id'].unique())
        cntbytrial=pd.Series(0,pd.MultiIndex.from_product([tmp['id'].unique(),range(n)], names=['id', 'state']))
        cntbytrial=pd.DataFrame({'null':cntbytrial,'cnt':tmp.groupby(['id','state']).count()[1]}).fillna(0)['cnt']
        cntbytrial=(cntbytrial/cntbytrial.groupby(level=0).sum())
        std[style]=cntbytrial.groupby(level=1).std()
        duration_se.loc[:,style]=std[style]/np.sqrt(ndf[style])
    se = np.sqrt(std['INS']*std['INS']/ndf['INS']+std['ANA']*std['ANA']/ndf['ANA'])
    zscore['proportion']=(duration['INS']-duration['ANA'])/se
    duration=duration.stack()
    duration_se=duration_se.stack()
    visitlen=pd.DataFrame()
    visitlen_se=pd.DataFrame()
    for style in ['INS','ANA','TO']:
        states=df[df['style']==style].groupby(['state','chgflag']).count()[1]
        visitlen.loc[:,style]=stepsize*states.unstack(level=1).mean(axis=1)
        ndf[style]=states.groupby('state').count()
        std[style]=stepsize*states.unstack(level=1).std(axis=1)
        visitlen_se.loc[:,style]=std[style]/np.sqrt(ndf[style])
    se = np.sqrt(std['INS']**2/ndf['INS']+std['ANA']**2/ndf['ANA'])
    zscore['visitlen']=(visitlen['INS']-visitlen['ANA'])/se
    visitlen=visitlen.stack()
    visitlen_se=visitlen_se.stack()
    visitfreq=pd.DataFrame()
    visitfreq_se=pd.DataFrame()
    for style in ['INS','ANA','TO']:
        states=df[df['style']==style].groupby(['state','chgflag']).count()[1]
        total=states.sum()
        visitfreq.loc[:,style]=[len(states.loc[i])/total for i in range (n)]
        ndf[style]=total
        visitfreq_se.loc[:,style]=np.sqrt(visitfreq[style]*(1-visitfreq[style])/ndf[style])
    p=(visitfreq['INS']*ndf['INS']+visitfreq['ANA']*ndf['ANA'])/(ndf['INS']+ndf['ANA'])
    se = np.sqrt(p * ( 1 - p ) * [ (1/ndf['INS']) + (1/ndf['ANA']) ])
    zscore['visitfreq']=(visitfreq['INS']-visitfreq['ANA'])/se
    visitfreq=visitfreq.stack()
    visitfreq_se=visitfreq_se.stack()
    return pd.DataFrame({'proportion':duration,'visitlen':visitlen,'visitfreq':visitfreq}),\
            pd.DataFrame({'proportion':duration_se,'visitlen':visitlen_se,'visitfreq':visitfreq_se}),zscore