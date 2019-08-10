import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa as lb
from scipy.io import wavfile as wav
from python_speech_features import mfcc,logfbank
import IPython.display as ipd

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def main():
    Path="C:\\Users\\sarth\\Desktop\\datasets\\UrbanSound8K\\metadata"
    f=os.path.join(Path,'UrbanSound8K.csv')
    df=pd.read_csv(f)
    df.set_index('slice_file_name',inplace=True)
    for i in df.index:
        df.at[i,'length']=df.at[i,'end']-df.at[i,'start']
    df.rename(columns={'class':'label'},inplace=True)    
    classes=list(np.unique(df.label))        
    class_dictionary=df.groupby(['label'])['length'].mean()
    
    fig,ax=plt.subplots()
    ax.set_title('Class Distribution',y=1.08)
    ax.pie(class_dictionary,labels=class_dictionary.index,autopct='%1.1f%%',shadow=False,startangle=90)
    ax.axis('equal')
    plt.show()
    df.reset_index(inplace=True)
    
    signals= {}
    mfccs= {}
    fft= {}
    fbank= {}
    
    def envelope(signal,sr,threshold):
        mask=[]
        y=pd.Series(signal).apply(np.abs)
        y_mean=y.rolling(window=int(sr/10),min_periods=1,center=True).mean()
        for mean in y_mean:
            if mean > threshold:
                mask.append(True)
            else:
                mask.append(False)
        return mask
    
    def calc_fft(y,rate):
        n=len(y)
        freq=np.fft.rfftfreq(n,d=1./rate)
        Y=abs(np.fft.rfft(y)/n)
        return (Y,freq)

    for i in classes:
       file=df[df.label==i].iloc[1][0]
       fold=df[df.label==i].iloc[1][5]
       sig,rate=lb.load('audio/fold'+str(fold)+'/'+file,sr=44100)
       mask=envelope(sig,rate,threshold=0.0005)
       signal=sig[mask]
       signals[i]=signal
       fft[i]=calc_fft(signal,rate)
       bank=logfbank(signal[:rate],rate,nfilt=26,nfft=1103)
       fbank[i]=bank
       melc=mfcc(signal[:rate],rate,numcep=13,nfilt=26,nfft=1103)
       mfccs[i]=melc
       
       
    plot_signals(signals)
    plt.show()
    
    plot_fft(fft)
    plt.show()
    
    plot_mfccs(mfccs)
    plt.show()
    
    plot_fbank(fbank)
    plt.show()


    if len(os.listdir('cleandata'))==0:
        for i in tqdm(df.index):
            s,rate=lb.load('audio/fold'+str(df.at[i,'fold'])+'/'+str(df.at[i,'slice_file_name']),sr=16000)
            mask=envelope(s[:rate],rate,0.0005)
            wav.write(filename='cleandata/'+str(df.at[i,'slice_file_name']),rate=rate,data=s[:rate])
            
            
if __name__ == "__main__":
    main()           




