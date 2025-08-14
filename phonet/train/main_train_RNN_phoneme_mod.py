import sys
from six.moves import cPickle as pickle

from keras.layers import Input, BatchNormalization, Bidirectional, GRU, Permute, Reshape, Lambda, Dense, RepeatVector, multiply, TimeDistributed, Dropout, LSTM
from keras.utils import np_utils
from keras.models import Model
from keras import optimizers
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import os
from utils import plot_confusion_matrix
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
#import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

from utils import confusion_matrix, get_scaler
from Phonological import Phonological

Phon=Phonological()


def generate_data(directory, batch_size, problem, mu, std, num_labels):
    i = 0
    file_list = os.listdir(directory)
    file_list.sort()
    while True:
        seq_batch = []
        y=[]
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                np.random.shuffle(file_list)
            with open(directory+file_list[i], 'rb') as f:
                save = pickle.load(f)
            f.close()
            seq_batch.append((save['features']-mu)/std)
            y.append(save['labels'][problem])
            i += 1
        y=np.stack(y, axis=0)
        
        y2=np_utils.to_categorical(y)
        ystack=np.concatenate(y, axis=0)
        ystack=np.hstack(ystack)
        lab, count=np.unique(ystack, return_counts=True)

        class_weights=class_weight.compute_class_weight('balanced', np.unique(ystack), ystack)

        weights=np.zeros((y.shape))

        for j in range(len(lab)):
            p=np.where(y==lab[j])
            weights[p]=class_weights[j]

        if np.max(y)<num_labels-1:
            da=np.zeros((batch_size, y2.shape[1], num_labels))
            da[:,:,0:np.max(y)+1]=y2
            y2=da

        seq_batch=np.stack(seq_batch, axis=0)
        yield seq_batch, y2, weights[:,:,0]


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

def get_test_labels(directory_cache, directory, problem):
    labels_cache_path = directory_cache + "test_labels_cache.npy"
    if os.path.exists(labels_cache_path):
        return np.load(labels_cache_path)
    
    file_list = os.listdir(directory)
    file_list.sort()
    total_files = len(file_list)
    
    
    y = []
    for i in range(total_files):
        if (i + 1) % 100 == 0 or (i + 1) % max(1, total_files // 10) == 0:
            percentage = ((i + 1) / total_files) * 100
            print(f"Progress: {i + 1}/{total_files} files ({percentage:.1f}%)")
        
        with open(directory + file_list[i], 'rb') as f:
            save = pickle.load(f)
        f.close()
        y.append(save['labels'][problem])
    
    
    y = np.stack(y, axis=0)
    ystack = np.concatenate(y, axis=0)
    
    return ystack
    
def DeepArch(input_size, GRU_size, hidden_size, num_labels, Learning_rate, recurrent_droput_prob):
    input_data=Input(shape=(input_size))
    x=input_data
    x=BatchNormalization()(x)
    x=Bidirectional(GRU(GRU_size, recurrent_dropout=recurrent_droput_prob, return_sequences=True))(x)
    x=Bidirectional(GRU(GRU_size, recurrent_dropout=recurrent_droput_prob, return_sequences=True))(x)
    x=Dropout(0.2)(x)
    x = TimeDistributed(Dense(hidden_size, activation='relu'))(x)
    x=Dropout(0.2)(x)
    x = TimeDistributed(Dense(num_labels, activation='softmax'))(x)
    modelGRU=Model(inputs=input_data, outputs=x)
    opt=optimizers.Adam(lr=Learning_rate)
    modelGRU.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'], sample_weight_mode="temporal")
    return modelGRU


if __name__=="__main__":

    if len(sys.argv)!=4:
        print("python main_train_RNN_phoneme_mod.py <path_seq_train> <path_seq_test> <path_results>")
        sys.exit()

    file_feat_train=sys.argv[1]
    file_feat_test=sys.argv[2]
    file_results=sys.argv[3]
    problem="phoneme_code"

    Nfiles_train=len(os.listdir(file_feat_train))
    Nfiles_test=len(os.listdir(file_feat_test))

    if not os.path.exists(file_results):
        os.makedirs(file_results)

    weights_h5_path = file_results + problem + '.h5'
    weights_hdf5_path = file_results + 'phonemes_weights.hdf5'
    model_json_path = file_results + problem + ".json"

    if os.path.exists(file_results+"mu.npy"):
        mu=np.load(file_results+"mu.npy")
        std=np.load(file_results+"std.npy")
    else:
        mu, std=get_scaler(file_feat_train)
        np.save(file_results+"mu.npy", mu)
        np.save(file_results+"std.npy", std)

    phonemes=Phon.get_list_phonemes()
    input_size=(40,34)
    GRU_size=128
    hidden=128
    num_labels=len(phonemes)
    Learning_rate=0.0005
    recurrent_droput_prob=0.0
    epochs=5
    batch_size=64

    modelPH=DeepArch(input_size, GRU_size, hidden, num_labels, Learning_rate, recurrent_droput_prob)
    
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
        steps_per_epoch=int(Nfiles_train/batch_size)
        validation_steps=int(Nfiles_test/batch_size)

        earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
        history=modelPH.fit_generator(generate_data(file_feat_train, batch_size, problem, mu, std, num_labels), 
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=epochs, shuffle=True, 
                                    validation_data=generate_data(file_feat_test, batch_size, problem, mu, std, num_labels), 
                                    verbose=1, callbacks=[earlystopper, checkpointer], 
                                    validation_steps=validation_steps)

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

    np.set_printoptions(precision=4)
    batch_size_val=1
    validation_steps=int(Nfiles_test/batch_size_val)

    ypred=modelPH.predict_generator(generate_data_test(file_feat_test, batch_size_val, mu, std), steps=validation_steps)

    ypredv=np.argmax(ypred, axis=2)
    ypredv=np.concatenate(ypredv, axis=0)

    yt=get_test_labels(file_results ,file_feat_test, problem)

    labels_cache_path = file_results + "test_labels_cache.npy"
    print(f"Saving processed labels to cache: {labels_cache_path}")
    np.save(labels_cache_path, yt)

    ytv=np.concatenate(yt,0)
    print(ytv.shape, ypredv.shape, ypred.shape)

    
    
    try:
        dfclass = classification_report(ytv, ypredv, target_names=phonemes, digits=4)
        print(dfclass)
    except Exception as e:
        dfclass = classification_report(ytv, ypredv, digits=4)
        print(dfclass)

    try:
        ax2 = plot_confusion_matrix(ytv, ypredv, file_res=file_results+"/cm.png", 
                                  classes=phonemes, normalize=True,
                                  title='Confusion Matrix - Phonemes')
    except Exception as e:
        all_classes = np.unique(np.concatenate([ytv, ypredv]))
        present_phonemes = [phonemes[i] if i < len(phonemes) else f"Class_{i}" 
                           for i in range(len(all_classes))]
        try:
            ax2 = plot_confusion_matrix(ytv, ypredv, file_res=file_results+"/cm.png", 
                                      classes=present_phonemes, normalize=True,
                                      title='Confusion Matrix - Present Phonemes')
        except Exception as e2:
            ax2 = plot_confusion_matrix(ytv, ypredv, file_res=file_results+"/cm.png", 
                                      normalize=True, title='Confusion Matrix')

    prec=precision_score(ytv, ypredv, average='weighted')
    rec=recall_score(ytv, ypredv, average='weighted')
    f1=f1_score(ytv, ypredv, average='weighted')

    F=open(file_results+"params.csv", "w")
    header="acc_train, acc_dev, loss, val_loss, epochs_run, Fscore, precision, recall\n"
    
    if history is not None:
        content=str(history.history["categorical_accuracy"][-1])+", "+str(history.history["val_categorical_accuracy"][-1])+", "
        content+=str(history.history["loss"][-1])+", "+str(history.history["val_loss"][-1])+", "
        content+=str(len(history.history["loss"]))+", "+str(f1)+", "+str(prec)+", "+str(rec)
    else:
        content=f"N/A, N/A, N/A, N/A, N/A, {f1}, {prec}, {rec}"
    
    F.write(header)
    F.write(content)
    F.close()