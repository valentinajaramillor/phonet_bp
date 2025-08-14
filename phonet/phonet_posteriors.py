
# -*- coding: utf-8 -*-
"""
Created on Jun 24 2020

@author: J. C. Vasquez-Correa
"""

"""
Modified for Phonet adaptation to Brazilian Portuguese
"""

import os
import sys

import numpy as np
import pandas as pd

from phonet import Phonet
from phonet import Phonological as phon

import scipy.stats as st
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(PATH, '..'))
sys.path.append(PATH)

#from disvoice_utils import save_dict_kaldimat, get_dict
from disvoice_utils import get_dict

from script_manager import script_manager
import torch
from tqdm import tqdm

class Phonological:
    """
    Compute phonological features from continuous speech files.

    It computes the phonological log-likelihood ratio features from phonet

    Script is called as follows

    >>> python phonet_posteriors.py <file_or_folder_audio> <file_features> <static (true or false)> <plots (true or false)> <format (csv, txt, npy, torch)>

    Examples command line:

    >>> python phonet_posteriors.py audios\ examples_results\phonologicalfeaturesst.csv true false csv

    """

    def __init__(self):
        phonolist=phon()
        self.head_dyn=phonolist.get_list_phonological_keys()
        self.statistics=["mean", "std", "skewness", "kurtosis", "max", "min"]
        self.head_st=[]
        for j in self.head_dyn:
            for l in self.statistics:
                self.head_st.append(j+"_"+l)
        self.phon=Phonet(["all"])

    def extract_features_file(self, audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """Extract the phonological features from an audio file

        :param audio: .wav audio file.
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldi features, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> phonological=Phonological()
        >>> file_audio="../audios/001_ddk1_PCGITA.wav"
        >>> features1=phonological.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
        >>> features2=phonological.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
        >>> features3=phonological.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
        >>> phonological.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")

        >>> phonological=Phonological()
        >>> path_audio="../audios/"
        >>> features1=phonological.extract_features_path(path_audio, static=True, plots=False, fmt="npy")
        >>> features2=phonological.extract_features_path(path_audio, static=True, plots=False, fmt="csv")
        >>> features3=phonological.extract_features_path(path_audio, static=False, plots=True, fmt="torch")
        >>> phonological.extract_features_path(path_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")

        """
        if static and fmt=="kaldi":
            raise ValueError("Kaldi is only supported for dynamic features")

        if audio.find('.wav') == -1 and audio.find('.WAV') == -1:
            raise ValueError(audio+" is not a valid wav file")
        
        df=self.phon.get_PLLR(audio, plot_flag=plots)

        keys=df.keys().tolist()
        keys.remove('time')

        if static:
            feat=[]
            functions=[np.mean, np.std, st.skew, st.kurtosis, np.max, np.min]
            for j in keys:
                for function in functions:
                    feat.append(function(df[j]))
            feat=np.expand_dims(feat, axis=0)

        else:
            feat=np.stack([df[k] for k in keys], axis=1)

        if fmt in("npy","txt"):
            return feat
        elif fmt in("dataframe","csv") and static:
            dff = {}
            for e, k in enumerate(self.head_st):
                dff[k] = feat[:, e]
            return pd.DataFrame(dff)
        elif fmt in("dataframe","csv") and not static:
            return df
        elif fmt=="torch":
            return torch.from_numpy(feat)
        
        else:
            raise ValueError(fmt+" is not supported")



    def extract_features_path(self, path_audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """Extract the phonological features for audios inside a path
        
        :param path_audio: directory with (.wav) audio files inside, sampled at 16 kHz
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldifeatures, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> phonological=Phonological()
        >>> path_audio="../audios/"
        >>> features1=phonological.extract_features_path(path_audio, static=True, plots=False, fmt="npy")
        >>> features2=phonological.extract_features_path(path_audio, static=True, plots=False, fmt="csv")
        >>> features3=phonological.extract_features_path(path_audio, static=False, plots=True, fmt="torch")
        >>> phonological.extract_features_path(path_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")
        """

        hf=os.listdir(path_audio)
        hf.sort()

        pbar=tqdm(range(len(hf)))
        ids=[]

        Features=[]
        for j in pbar:
            pbar.set_description("Processing %s" % hf[j])
            audio_file=path_audio+hf[j]
            feat=self.extract_features_file(audio_file, static=static, plots=plots, fmt="npy")
            Features.append(feat)
            if static:
                ids.append(hf[j])
            else:
                ids.append(np.repeat(hf[j], feat.shape[0]))
        
        Features=np.vstack(Features)
        ids=np.hstack(ids)

        return self.save_features(Features, ids, fmt, static, kaldi_file)



    def save_features(self, Features, ids, fmt, static, kaldi_file):

        if static:
            head = self.head_st
        else:
            head = self.head_dyn
    
        if fmt in("npy","txt"):
            return Features
        elif fmt in("dataframe","csv"):
            df = {}
            for e, k in enumerate(head):
                df[k] = Features[:, e]
            df["id"] = ids
            return pd.DataFrame(df)
        elif fmt=="torch":
            return torch.from_numpy(Features)
        elif fmt=="kaldi":
            dictX=get_dict(Features, ids)
            save_dict_kaldimat(dictX, kaldi_file)
        else:
            raise ValueError(fmt+" is not supported")


if __name__=="__main__":

    print(len(sys.argv))
    print(sys.argv)
    
    if len(sys.argv)!=6:
        print("python phonet_posteriors.py <file_or_folder_audio> <file_features> <static (true or false)> <plots (true or false)> <format (csv, txt, npy, torch)>")
        sys.exit()

    phonological=Phonological()
    script_manager(sys.argv, phonological)