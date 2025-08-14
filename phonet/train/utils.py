
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from six.moves import cPickle as pickle
from tqdm import tqdm
from Phonological import Phonological

Phon=Phonological()

def test_labels(directory):

    file_list = os.listdir(directory)
    file_list.sort()
    with open(directory+file_list[0], 'rb') as f:
        save = pickle.load(f)
    f.close()
    seq=save['labels']
    keys=Phon.get_list_phonological_keys()

    pbar=tqdm(range(len(file_list)))
    percall=np.zeros(len(keys))
    percall2=np.zeros(len(keys))

    for j in pbar:
        pbar.set_description("Processing %s" % file_list[j])

        with open(directory+file_list[j], 'rb') as f:
            save = pickle.load(f)
        f.close()
        seq=save['labels']
        perc1=np.zeros(len(keys))
        for e, k in enumerate(keys):
            perc1[e]=np.mean(seq[k])
        percall+=perc1
        percall2+=perc1

    percall=percall/len(file_list)

    return percall


def get_scaler(directory):
    i = 0
    file_list = os.listdir(directory)
    file_list.sort()
    seq_sum=np.zeros((40,34))
    seq_std=np.zeros((40,34))
    pbar=tqdm(range(len(file_list)))
    nans=0
    infs=0
    for j in pbar:
        pbar.set_description("Processing %s" % file_list[j])
        with open(directory+file_list[j], 'rb') as f:
            save = pickle.load(f)
        f.close()
        seq=save['features']
        seq_sum+=seq
        if np.sum(np.isnan(seq))>0:
            nans+=1
        if np.sum(np.isinf(seq))>0:
            infs+=1
    N=len(file_list)
    mu=seq_sum/N
    
    print("--------------------------")
    print("NAN", nans)
    print("INF", infs)
    pbar2=tqdm(range(len(file_list)))
    for j in pbar2:
        pbar2.set_description("Processing %s" % file_list[j])
        with open(directory+file_list[j], 'rb') as f:
            save = pickle.load(f)
        f.close()
        seq=save['features']
        seq_std+=(seq-mu)**2

    std=seq_std/len(file_list)

    find0=np.where(std==0)[0]
    print("std0", len(find0))

    return mu, std

def plot_confusion_matrix2(y_true, y_pred, classes, file_res, 
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #print(unique_labels(y_true, y_pred))
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    np.set_printoptions()
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True phoneme',
           xlabel='Predicted phoneme')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.0f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(100*cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(file_res)
    return ax

def plot_confusion_matrix(y_true, y_pred, classes, file_res, 
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          figsize=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    np.set_printoptions()
    print(cm)

    # Determinar tamaño de figura automáticamente basado en número de clases
    num_classes = len(classes)
    if figsize is None:
        if num_classes <= 10:
            figsize = (8, 6)
        elif num_classes <= 20:
            figsize = (12, 10)
        elif num_classes <= 30:
            figsize = (16, 14)
        else:
            # Para 46 fonemas o más
            figsize = (20, 18)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Ajustar tamaño de fuente basado en número de clases
    if num_classes <= 20:
        label_fontsize = 8
        tick_fontsize = 8
        text_fontsize = 8
    elif num_classes <= 30:
        label_fontsize = 6
        tick_fontsize = 6
        text_fontsize = 6
    else:
        # Para 46 fonemas
        label_fontsize = 5
        tick_fontsize = 5
        text_fontsize = 4
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True phoneme',
           xlabel='Predicted phoneme')

    # Configurar tamaños de fuente
    ax.title.set_fontsize(12)
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    
    # Rotar etiquetas y ajustar tamaño
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=tick_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=tick_fontsize)

    # Mostrar TODOS los números en todas las celdas
    fmt = '.0f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(100*cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=text_fontsize)
    
    plt.tight_layout()
    plt.savefig(file_res, dpi=300, bbox_inches='tight')
    plt.close()  # Cerrar figura para liberar memoria
    return ax
