import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa as lb
import IPython.display as ipd
from cfg import config
import random
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dropout, MaxPooling2D, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

#for efficient GPU use
import tensorflow as tf
from keras.backend import tensorflow_backend
conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=conf)
tensorflow_backend.set_session(session)

from sklearn.utils.class_weight import compute_class_weight

config=config()
Path="C:\\Users\\sarth\\Desktop\\datasets\\UrbanSound8K\\metadata"
f=os.path.join(Path,'UrbanSound8K.csv')
df=pd.read_csv(f)
df.rename(columns={'class':'label'},inplace=True)
classes=list(np.unique(df.label))

x_tr,x_te,y_tr,y_te=[],[],[],[]

def features():
    x,xt=[],[]
    for i in tqdm(range(len(df))):
    #for i in range(3):
        fold=df.iloc[i]['fold']
        file_name=df.iloc[i]['slice_file_name']
        label_name=df.iloc[i]['label']
        signal,rate=lb.load('audio/fold'+str(fold)+'/'+file_name)
        mfccs = np.mean(lb.feature.mfcc(signal, rate, n_mfcc=40).T,axis=0)
        melspec = np.mean(lb.feature.melspectrogram(y=signal, sr=rate, n_mels=40,fmax=8000).T,axis=0)
        chroma_stft = np.mean(lb.feature.chroma_stft(y=signal, sr=rate,n_chroma=40).T,axis=0)
        chroma_cq = np.mean(lb.feature.chroma_cqt(y=signal, sr=rate,n_chroma=40).T,axis=0)
        chroma_cens = np.mean(lb.feature.chroma_cens(y=signal, sr=rate,n_chroma=40).T,axis=0)
        featureset = np.reshape(np.vstack((mfccs,melspec,chroma_stft,chroma_cq,chroma_cens)),(40,5))
        
        if (fold!=10):
            x_tr.append(featureset)
            y_tr.append(classes.index(label_name))
        else:
            x_te.append(featureset)
            y_te.append(classes.index(label_name))
     
    x.append(x_tr)
    #y.append(y_tr)
    xt.append(x_te)
    #yt.append(y_te)
    x,y,x_t,y_t=np.array(x),np.array(y_tr),np.array(xt),np.array(y_te)
    x_train=(x-np.mean(x))/np.std(x)
    x_test=(x_t-np.mean(x_t))/np.std(x_t)      
    y_train=to_categorical(y,num_classes=10)
    y_test=to_categorical(y_t,num_classes=10)
    return (x_train,y_train,x_test,y_test)       

def conv():
    model=Sequential()
    
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1),padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    #model.add(Dropout(0.3))
    
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    #model.add(Dropout(0.3))
    
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    #model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding='same'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.3))
    
    model.add(Flatten())
    
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

if config.mode=='conv':
    x_train,y_train,x_test,y_test=features()
    y_feat=np.argmax(y_train,axis=1)
    #input_shape=(x_train.shape[0],x_train.shape[1],1)
    input_shape=(40,5,1)
    x_train = np.reshape(x_train, (x_train.shape[1], 40, 5, 1))
    x_test = np.reshape(x_test, (x_test.shape[1], 40, 5, 1))
    
    model=conv()    
else:
    print('BLNT')


class_weight=compute_class_weight('balanced',np.unique(y_feat),y_feat)  
checkpoint=ModelCheckpoint(config.model_path,monitor='val_acc',verbose=1,mode='max',save_best_only=True,save_weights_only=False,period=1)
history=model.fit(x_train,y_train,epochs=100,batch_size=32,verbose=1,validation_split=0.1,validation_data=(x_test,y_test))

import matplotlib.pyplot as plt
#plt.figure(figsize=(5,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])  
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') 
plt.show()


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])   
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
    
    
    