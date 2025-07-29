
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score,roc_curve,auc

def plot_history(histories, key='binary_crossentropy'):
#def plot_history(histories, key='acc'):
    # plt.figure(figsize=(16,8))
    plt.grid()
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val',color='red')
        plt.plot(history.epoch, history.history[key], color='black',
             label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.title("Training History")
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    # plt.show()
    plt.savefig('pdf/nn/training_history-{}.pdf'.format(key))
    plt.close()

def  plot_sig_bkg_from_np_arrays(features,labels,feature_names,logy=False):
    """Plot sig and bkg"""
    #use boolean mask index for specified signal/bkg, sig=1.
    sig=features[labels==1.0]
    bkg=features[labels==0.0]

    num_features = sig.shape[1]
    sig_list = np.split(sig, num_features, axis=1)
    bkg_list = np.split(bkg, num_features, axis=1)

    plt.figure(figsize=(18,10))
    for i in range(num_features):
    #for i in range(4):
        plt.subplot(3,6,i+1)
        #plt.xticks([])
        plt.yticks([])
        plt.grid(True)
        if logy:
            plt.yscale('log')
        plt.hist(sig_list[i],density=True,bins=100,histtype='step',label="Signal",color='r')
        plt.hist(bkg_list[i],density=True,bins=100,histtype='step',label="Bkg",color='b')
        plt.xlabel(feature_names[i])
    # plt.show()
    plt.legend()
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0
    plt.savefig('pdf/dist-sig-bkg.pdf',dpi=200)
    plt.close()

def plot_roc(label,score):
    path = "pdf/nn/"

    fake, true, _ = roc_curve(label, score)
    roc_auc = auc(fake,true)
    plt.figure(figsize=[6, 6])
    lw=2
    plt.plot(fake, true, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(path+'roc-curve.pdf',dpi=200)

def plot_score(label,score,logy=False):
    score_bkg = score[np.where(label == 0)]
    score_sig = score[np.where(label == 1)]
    path = "pdf/nn/"

    plt.figure(figsize=[6, 6])
    if logy:
        plt.yscale('log')
    plt.hist([score_bkg, score_sig],label=["Bkg","Sig"],density=True,bins=30, histtype='step',fill=True, alpha=0.3)
    plt.xlabel('Score')
    plt.title("Training Score Sig/Bkg Plot")
    plt.ylabel("Fraction")
    plt.legend()
    plt.savefig(path+'ml-score.pdf')

    
def comp_plot(ptitle,vnames,reald,generd,inputd=None,logy=False):

    #histo limits
    f_max=np.amax(reald,axis=0)
    f_min=np.amin(reald,axis=0)
    feat_dim=reald.shape[1]

    for i in range(feat_dim):
        if not i%6:
            plt.figure(figsize=(16,8))
            plt.suptitle(ptitle)
        plt.subplot(2,3,i%6+1)
        #ax=plt.subplot(2,3,i%6+1)
        #ax.set_xlim([-10.,10.])
        ax_range=(f_min[i],min(f_max[i],5.))
        #ax_range=(-6.,6.)
        #plt.xticks([])
        plt.yticks([])
        plt.grid(True)
        plt.hist(reald[:,i],density=True,bins=100,histtype='step',color='b',label="Real",range=ax_range)
        plt.hist(generd[:,i],density=True,bins=100,histtype='step',color='r',label="Generated",range=ax_range)
        if inputd is not None:
            plt.hist(inputd[:,i],density=True,bins=100,histtype='step',color='g',label="Latent",range=ax_range)
        plt.xlabel(vnames[i])
        plt.legend()
        if (not (i+1)%6 and i != 0) or (i+1 == feat_dim):
            plt.savefig('bvae_feat-{}-{}.png'.format(ptitle.lower().replace(" ","-"),i))
            plt.close()

def comp_plot_dict(ptitle,vnames,ddict,logy=False):

    #histo limits from first entry
    reald= list(ddict.values())[0]
    f_max=np.amax(reald,axis=0)
    f_min=np.amin(reald,axis=0)
    feat_dim=reald.shape[1]

    pcolors=['b','r','g','y']
    for i in range(feat_dim):
        if not i%6:
            plt.figure(figsize=(16,8))
            plt.suptitle(ptitle)
        plt.subplot(2,3,i%6+1)
        ax_range=(f_min[i],f_max[i])
        #ax_range=(max(-5.,f_min[i]),min(f_max[i],5.))
        #plt.xticks([])
        plt.yticks([])
        plt.grid(True)
        #plt.xscale('log')
        if logy:
            plt.yscale('log')
        nit=0
        for dlabel,ddata in ddict.items():
            plt.hist(ddata[:,i],density=True,bins=100,histtype='step',color=pcolors[nit],label=dlabel,range=ax_range)
            nit +=1
        plt.xlabel(vnames[i])
        plt.legend()
        if (not (i+1)%6 and i != 0) or (i+1 == feat_dim):
            plt.savefig('bvae_feat-{}-{}.png'.format(ptitle.lower().replace(" ","-"),i))
            plt.close()

def comp_plot_dict_overlay(ptitle,vnames,ddict,logy=False):

    #histo limits from first entry
    reald= list(ddict.values())[0]
    f_max=np.amax(reald,axis=0)
    f_min=np.amin(reald,axis=0)
    feat_dim=reald.shape[1]

    plt.figure(figsize=(16,8))
    plt.suptitle(ptitle)
    #ax_range=(f_min[0],f_max[0])
    ax_range=(np.amin(f_min),np.amax(f_max))
    for i in range(feat_dim):
        plt.yticks([])
        plt.grid(True)
        #plt.xscale('log')
        if logy:
            plt.yscale('log')
        nit=0
        for dlabel,ddata in ddict.items():
            plt.hist(ddata[:,i],density=True,bins=100,histtype='step',label=dlabel,range=ax_range)
            nit +=1
    plt.xlabel(vnames[0])
    plt.legend()
    plt.savefig('bvae_feat-{}-{}.png'.format(ptitle.lower().replace(" ","-"),"overlay"))
    plt.close()

def cumulant_plot_dict(ptitle,vnames,ddict,logy=False):

    #histo limits from first entry
    reald= list(ddict.values())[0]
    feat_dim=reald.shape[1]

    pcolors=['b','r','g','y']
    for i in range(feat_dim):
        if not i%6:
            plt.figure(figsize=(16,8))
            plt.suptitle(ptitle)
        plt.subplot(2,3,i%6+1)
        #ax=plt.subplot(2,3,i%6+1)
        #plt.xticks([])
        #plt.yticks([])
        plt.grid(True)
        if logy:
            plt.yscale('log')
        nit=0
        for dlabel,ddata in ddict.items():
            # sort the data:
            data_sorted = np.sort(ddata[:,i])
            # calculate the proportional values of samples
            #print("sample {} size {}".format(nit,data_sorted[0:3]))
            p = 1. * np.arange(len(data_sorted)) / (len(data_sorted) - 1)
            plt.plot(data_sorted,p,'-{}'.format(pcolors[nit]),label=dlabel)
            nit +=1
        plt.xlabel(vnames[i])
        plt.legend()
        if (not (i+1)%6 and i != 0) or (i+1 == feat_dim):
            plt.savefig('bvae_feat-{}-{}.png'.format(ptitle.lower().replace(" ","-"),i))
            plt.close()

def cumhist_plot_dict(ptitle,vnames,ddict,logy=False):

    #histo limits from first entry
    reald= list(ddict.values())[0]
    f_max=np.amax(reald,axis=0)
    f_min=np.amin(reald,axis=0)
    feat_dim=reald.shape[1]

    pcolors=['b','r','g','y']
    for i in range(feat_dim):
        if not i%6:
            plt.figure(figsize=(16,8))
            plt.suptitle(ptitle)
        plt.subplot(2,3,i%6+1)
        #ax=plt.subplot(2,3,i%6+1)
        ax_range=(f_min[i],min(f_max[i],5.))
        #plt.xticks([])
        #plt.yticks([])
        plt.grid(True)
        if logy:
            plt.yscale('log')
        nit=0
        for dlabel,ddata in ddict.items():
            plt.hist(ddata[:,i],density=True,cumulative=True,bins=1000,histtype='step',color=pcolors[nit],label=dlabel,range=ax_range)
            nit +=1
        plt.xlabel(vnames[i])
        plt.legend()
        if (not (i+1)%6 and i != 0) or (i+1 == feat_dim):
            plt.savefig('bvae_feat-{}-{}.png'.format(ptitle.lower().replace(" ","-"),i))
            plt.close()


def cumdiff_plot_dict(ptitle,vnames,ddict):

    nbins=1000
    #histo limits from first entry, needs two entries only...
    reald= list(ddict.values())[0]
    f_max=np.amax(reald,axis=0)
    f_min=np.amin(reald,axis=0)
    feat_dim=reald.shape[1]


    pcolors=['b','r','g','y']

    Alab= list(ddict.keys())[0]
    Avals= list(ddict.values())[0]
    Asum= float(Avals.shape[0])
    Blab= list(ddict.keys())[-1]
    Bvals= list(ddict.values())[-1]
    Bsum= float(Bvals.shape[0])

    for i in range(feat_dim):
        if not i%6:
            plt.figure(figsize=(16,8))
            plt.suptitle(ptitle)
        plt.subplot(2,3,i%6+1)
        ax_range=(f_min[i],min(f_max[i],5.))
        #plt.xticks([])
        #plt.yticks([])
        plt.grid(True)
        # Calculate cdfs of A and B
        Apdf, edges = np.histogram(Avals[:,i], bins=nbins, range=ax_range)
        x_val = (edges[0:-1] + edges[1:]) / 2
        Bpdf, edges = np.histogram(Bvals[:,i], bins=nbins, range=ax_range)
        Acdf = np.cumsum(Apdf) /Asum
        Bcdf = np.cumsum(Bpdf) /Bsum
        nit=0
        plt.plot(x_val,(Acdf-Bcdf),'-{}'.format(pcolors[nit]),label='{}-{}'.format(Alab,Blab))
        plt.xlabel(vnames[i])
        plt.legend()
        if (not (i+1)%6 and i != 0) or (i+1 == feat_dim):
            plt.savefig('bvae_feat-{}-{}.png'.format(ptitle.lower().replace(" ","-"),i))
            plt.close()

def cumul_and_diff_plot_dict(ptitle,vnames,ddict):

    nbins=1000
    #histo limits from first entry, needs two entries only...
    reald= list(ddict.values())[0]
    f_max=np.amax(reald,axis=0)
    f_min=np.amin(reald,axis=0)
    feat_dim=reald.shape[1]


    pcolors=['b','r','g','y']

    Alab= list(ddict.keys())[0]
    Avals= list(ddict.values())[0]
    Asum= float(Avals.shape[0])
    Blab= list(ddict.keys())[-1]
    Bvals= list(ddict.values())[-1]
    Bsum= float(Bvals.shape[0])

    for i in range(feat_dim):
        if not i%6:
            plt.figure(figsize=(16,8))
            plt.suptitle(ptitle)
        plt.subplot(2,3,i%6+1)
        ax_range=(f_min[i],min(f_max[i],5.))
        #plt.xticks([])
        #plt.yticks([])
        plt.grid(True)
        # Calculate cdfs of A and B
        Apdf, edges = np.histogram(Avals[:,i], bins=nbins, range=ax_range)
        x_val = (edges[0:-1] + edges[1:]) / 2
        Bpdf, edges = np.histogram(Bvals[:,i], bins=nbins, range=ax_range)
        Acdf = np.cumsum(Apdf) /Asum
        Bcdf = np.cumsum(Bpdf) /Bsum
        nit=0
        plt.plot(x_val,Acdf,'-{}'.format(pcolors[nit]),label='{}'.format(Alab))
        nit=1
        plt.plot(x_val,Bcdf,'-{}'.format(pcolors[nit]),label='{}'.format(Blab))
        nit=2
        plt.plot(x_val,(Acdf-Bcdf),'-{}'.format(pcolors[nit]),label='{}-{}'.format(Alab,Blab))
        plt.xlabel(vnames[i])
        plt.legend()
        if (not (i+1)%6 and i != 0) or (i+1 == feat_dim):
            plt.savefig('bvae_feat-{}-{}.png'.format(ptitle.lower().replace(" ","-"),i))
            plt.close()


def hist_and_rat_plot_dict(ptitle,vnames,ddict):

    nbins=100
    #histo limits from first entry, needs two entries only...
    reald= list(ddict.values())[0]
    f_max=np.amax(reald,axis=0)
    f_min=np.amin(reald,axis=0)
    feat_dim=reald.shape[1]


    pcolors=['b','r','g','y']

    Alab= list(ddict.keys())[0]
    Avals= list(ddict.values())[0]
    Asum= float(Avals.shape[0])
    Blab= list(ddict.keys())[-1]
    Bvals= list(ddict.values())[-1]
    Bsum= float(Bvals.shape[0])


    for i in range(feat_dim):
        if not i%6:
            fig0=plt.figure(figsize=(16,8))
            plt.suptitle(ptitle)
            outer = gridspec.GridSpec(2, 3, wspace=0.2, hspace=0.2)

        inner = outer[i%6].subgridspec(3, 1,wspace=0, hspace=0)

        ax = fig0.add_subplot(inner[0:2]) #main
        ax2 = fig0.add_subplot(inner[2]) #additional plot

        ax_range=(f_min[i],min(f_max[i],5.))
        #plt.xticks([])
        #plt.yticks([])
        plt.grid(True)
        # Calculate pdfs of A and B and plot both and ratio
        Apdf, edges = np.histogram(Avals[:,i], bins=nbins, range=ax_range)
        x_val = (edges[0:-1] + edges[1:]) / 2
        Bpdf, edges = np.histogram(Bvals[:,i], bins=nbins, range=ax_range)
        Acdf = np.array(Apdf) /Asum
        Bcdf = np.array(Bpdf) /Bsum
        nit=0
        ax.plot(x_val,Acdf,'-{}'.format(pcolors[nit]),label='{}'.format(Alab))
        nit=1
        ax.plot(x_val,Bcdf,'-{}'.format(pcolors[nit]),label='{}'.format(Blab))
        nit=2
        ax2.plot(x_val,(Bcdf/Acdf),'-{}'.format(pcolors[nit]),label='{}-{}'.format(Alab,Blab))
        ax2.set_ylim(0.5,1.5)
        ax2.set_xlabel(vnames[i])
        ax.legend()
        fig0.add_subplot(ax)
        fig0.add_subplot(ax2)
        if (not (i+1)%6 and i != 0) or (i+1 == feat_dim):
            plt.savefig('bvae_feat-{}-{}.png'.format(ptitle.lower().replace(" ","-"),i))
            plt.close()
