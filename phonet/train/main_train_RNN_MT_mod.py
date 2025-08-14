import sys
from six.moves import cPickle as pickle

from keras.layers import Input, BatchNormalization, Bidirectional, GRU, Permute, Reshape, Lambda, Dense, RepeatVector, multiply, TimeDistributed, Dropout, LSTM
from keras.utils import np_utils
from keras.models import Model
from keras import optimizers
import keras.backend as K
#from keras.callbacks import EarlyStopping, ModelCheckpoint, tensorboard_v1
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import os
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, confusion_matrix, balanced_accuracy_score
#import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils import plot_confusion_matrix

from sklearn.utils import class_weight

from utils import confusion_matrix, get_scaler, test_labels

from Phonological import Phonological

Phon=Phonological()




def generate_data(directory, batch_size, mu, std):
    i = 0
    file_list = os.listdir(directory)
    file_list.sort()
    keys=Phon.get_list_phonological_keys()
    while True:
        seq_batch = []
        
        y={}
        for k in keys:
            y[k]=[]

        class_weights=[]
        weights=[]
        y2=[]
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                np.random.shuffle(file_list)
            with open(directory+file_list[i], 'rb') as f:
                save = pickle.load(f)
            f.close()
            seq_batch.append((save['features']-mu)/std)

            for problem in keys:            
                y[problem].append(save['labels'][problem])
            i += 1

        for problem in keys:  
            ystack=np.stack(y[problem], axis=0)

            ystack2=np.concatenate(ystack, axis=0)
            ystack2=np.hstack(ystack2)
            lab, count=np.unique(ystack2, return_counts=True)
            class_weights.append(class_weight.compute_class_weight('balanced', np.unique(y[problem]), ystack2))
            weights_t=np.zeros((ystack.shape))
            for j in range(len(lab)):
                p=np.where(ystack==lab[j])
                weights_t[p]=class_weights[-1][j]
            weights.append(weights_t[:,:,0])
            if len(lab)>1:
                y2.append(np_utils.to_categorical(ystack))
            else:
                da=np.zeros((batch_size,ystack.shape[1], 2))
                da[:,:,0]=1
                y2.append(da)

        seq_batch=np.stack(seq_batch, axis=0)
        yield seq_batch, y2, weights


def generate_data_test(directory, batch_size, mu, std):
    i = 0
    file_list = os.listdir(directory)
    file_list.sort()
    while True:
        seq_batch = []
        for b in range(batch_size):
            with open(directory+file_list[i], 'rb') as f:
                save = pickle.load(f)
            f.close()
            seq_batch.append((save['features']-mu)/std)
            i+=1
        seq_batch=np.stack(seq_batch, axis=0)
        yield seq_batch




def get_test_labels(directory, batch_size):
    i = 0
    file_list = os.listdir(directory)
    file_list.sort()
    keys=Phon.get_list_phonological_keys()

    y={}
    for k in keys:
        y[k]=[]  

    for i in tqdm(range(len(file_list)), desc="Processing labels"):
        with open(directory+file_list[i], 'rb') as f:
            save = pickle.load(f)
        f.close()
        for problem in keys:
            y[problem].append(save['labels'][problem])
    
    for problem in keys:
        y[problem]=np.stack(y[problem], axis=0)
    return y



def DeepArch(input_size, GRU_size, hidden_size, num_labels, names, Learning_rate, recurrent_droput_prob):
    input_data=Input(shape=(input_size))
    x=input_data
    x=BatchNormalization()(x)
    x=Bidirectional(GRU(GRU_size, recurrent_dropout=recurrent_droput_prob, return_sequences=True))(x)
    x=Bidirectional(GRU(GRU_size, recurrent_dropout=recurrent_droput_prob, return_sequences=True))(x)
    x=Dropout(0.2)(x)
    x = TimeDistributed(Dense(hidden_size, activation='relu'))(x)
    x=Dropout(0.2)(x)
        # multi-task
    xout=[]
    out=[]
    for j in range(len(names)):
        xout.append(TimeDistributed(Dense(hidden_size, activation='relu'))(x))
        out.append(TimeDistributed(Dense(num_labels[j], activation='softmax'), name=names[j])(xout[-1]))

    modelGRU=Model(inputs=input_data, outputs=out)
    opt=optimizers.Adam(lr=Learning_rate)
    alphas=list(np.ones(len(names))/len(names))
    modelGRU.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'], sample_weight_mode="temporal", loss_weights=alphas)
    return modelGRU




if __name__=="__main__":

    if len(sys.argv)!=4:
        print("python main_train_RNN_MT_mod.py <path_seq_train> <path_seq_test> <path_results>")
        sys.exit()

    file_feat_train=sys.argv[1]
    file_feat_test=sys.argv[2]
    file_results=sys.argv[3]

    Nfiles_train=len(os.listdir(file_feat_train))
    Nfiles_test=len(os.listdir(file_feat_test))

    if not os.path.exists(file_results):
        os.makedirs(file_results)


    weights_h5_path = file_results + 'model.h5'
    weights_hdf5_path = file_results + 'weights.hdf5'
    model_json_path = file_results + "model.json"

    #perc=test_labels(file_feat_test)
    #print("perc_classes=", perc)



    if os.path.exists(file_results+'mu.npy'):

        mu=np.load(file_results+"mu.npy")
        std=np.load(file_results+"std.npy")
    else:
        mu, std=get_scaler(file_feat_train)

        np.save(file_results+"mu.npy", mu)
        np.save(file_results+"std.npy", std)

    input_size=(40,34)
    GRU_size=128
    hidden=128
    keys=Phon.get_list_phonological_keys()
    num_labels=[2 for j in range(len(keys))]
    Learning_rate=0.0001
    recurrent_droput_prob=0.0
    epochs=5
    batch_size=64


    modelPH=DeepArch(input_size, GRU_size, hidden, num_labels, keys, Learning_rate, recurrent_droput_prob)
        

    history = None
    
    if os.path.exists(weights_h5_path):
        print(f"Model weights found: {weights_h5_path}")
        print("Loading weights and skipping training...")
        modelPH.load_weights(weights_h5_path)
    elif os.path.exists(weights_hdf5_path):
        print(f"Model weights found: {weights_hdf5_path}")
        print("Loading weights and skipping training...")
        modelPH.load_weights(weights_hdf5_path)
    else:
        print("No trained model found. Starting training...")
        print(modelPH.summary())
        
        checkpointer = ModelCheckpoint(filepath=weights_hdf5_path, verbose=1, save_best_only=True)
        steps_per_epoch=int(Nfiles_train/batch_size)#
        validation_steps=int(Nfiles_test/batch_size)

        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        history=modelPH.fit_generator(generate_data(file_feat_train, batch_size, mu, std), steps_per_epoch=steps_per_epoch, epochs=epochs, shuffle=True, validation_data=generate_data(file_feat_test, batch_size, mu, std), 
        verbose=1, callbacks=[earlystopper, checkpointer], validation_steps=validation_steps)

        model_json = modelPH.to_json()
        with open(model_json_path, "w") as json_file:
            json_file.write(model_json)
        
        try:
            modelPH.save_weights(weights_h5_path)
        except:
            print(f"Could not save weights to {weights_h5_path}")

    
    if history is not None:
        plt.figure()
        plt.plot(np.log(history.history['loss']))
        plt.plot(np.log(history.history['val_loss']))
        plt.xlabel("epochs")
        plt.ylabel("log-Loss")
        plt.savefig(file_results+'Loss.png')
        plt.close('all')


    model_json = modelPH.to_json()
    with open(file_results+"model.json", "w") as json_file:
        json_file.write(model_json)
    try:
        modelPH.save_weights(file_results+'model.h5')
    except:
        print('------------------------------------------------------------------------------------------------------------')
        print('┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐')
        print('|                                                                                                           |')
        print('|      FILE  '+file_results+'.h5'+'                                      |')
        print('|             could not be saved                                                                            |')
        print('|                                                                                                           |')
        print('└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘')
        print('------------------------------------------------------------------------------------------------------------')
 


    np.set_printoptions(precision=4)
    batch_size_val=1
    validation_steps=int(Nfiles_test/batch_size_val)

    if os.path.exists(file_results+"ypred.npy"):
        ypred = np.load(file_results+"ypred.npy", allow_pickle=True)
    else:
        ypred=modelPH.predict_generator(generate_data_test(file_feat_test, batch_size_val, mu, std), steps=validation_steps)
        np.save(file_results+"ypred.npy", ypred)

    if os.path.exists(file_results + "test_labels_cache.pkl"):
        with open(file_results + "test_labels_cache.pkl", 'rb') as f:
            yt = pickle.load(f)
    else:
        yt=get_test_labels(file_feat_test, batch_size)
        with open(file_results + "test_labels_cache.pkl", 'wb') as f:
            pickle.dump(yt, f)


    F=open(file_results+"params.csv", "w")
    header="class, acc_train, acc_dev, loss, val_loss, epochs_run, Fscore, precision, recall, sensitivity, specificity, UAR\n"
    F.write(header)

    for e, problem in enumerate(keys):
        
        ypredv=np.argmax(ypred[e], axis=2)
        ypredv=np.concatenate(ypredv, axis=0)
        ytv=np.concatenate(yt[problem],0)

        print(ytv.shape, ypredv.shape)
        dfclass=classification_report(ytv, ypredv,digits=4)


        print(dfclass)

        class_names=["not "+problem, problem]

  
        ax2=plot_confusion_matrix(ytv, ypredv, file_res=file_results+problem+"cm.png", classes=class_names, normalize=True,
                        title='Normalized confusion matrix')

        prec=precision_score(ytv, ypredv, average='weighted')
        rec=recall_score(ytv, ypredv, average='weighted')
        f1=f1_score(ytv, ypredv, average='weighted')
        cm = confusion_matrix(ytv, ypredv)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        uar = balanced_accuracy_score(ytv, ypredv)

        if history is not None:
            content=problem+", "+str(history.history[problem+"_categorical_accuracy"][-1])+", "+str(history.history["val_"+problem+"_categorical_accuracy"][-1])+", "
            content+=str(history.history[problem+'_loss'][-1])+", "+str(history.history['val_'+problem+'_loss'][-1])+", "
            content+=str(len(history.history["loss"]))+", "+str(f1)+", "+str(prec)+", "+str(rec)+", "+str(sensitivity)+", "+str(specificity)+", "+str(uar)+'\n'
        else:
            content=problem+", N/A, N/A, N/A, N/A, N/A, "+str(f1)+", "+str(prec)+", "+str(rec)+", "+str(sensitivity)+", "+str(specificity)+", "+str(uar)+'\n'

    
        F.write(content)
    F.close()


