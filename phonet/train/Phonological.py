
"""
Created on Feb 28 2019
@author: J. C. Vasquez-Correa
        Pattern recognition Lab, University of Erlangen-Nuremberg
        Faculty of Engineering, University of Antioquia,
        juan.vasquez@fau.de
"""

"""
Modified for Brazilian Portuguese phonological classes
"""

import numpy as np
import pandas as pd

class Phonological:

    def __init__(self):

        self.list_phonological={
            "high"                 : ["i", "u", "1", "i~", "u~"],
            "upper_mid"            : ["e", "o", "e~", "o~"],
            "lower_mid"            : ["E", "O"],
            "low"                  : ["a", "6", "6~"],
            "front"                : ["i", "e", "E", "i~", "e~", "j", "j~"],
            "central"              : ["a", "6", "1", "6~"],
            "back"                 : ["u", "o", "O", "u~", "o~", "w", "w~"],
            "stop_voiceless"       : ["p", "t", "k", "c"],
            "stop_voiced"          : ["b", "d", "g", "J\\"],
            "fricative_voiceless"  : ["f", "s", "S", "x"],
            "fricative_voiced"     : ["v", "z", "Z", "D", "B", "G"],
            "nasal"                : ["m", "n", "J"],
            "trill"                : ["R\\"],
            "tap"                  : ["4"],
            "lateral"              : ["l", "L"],
            "bilabial"             : ["p", "b", "m", "B", "w", "w~"],
            "labiodental"          : ["f", "v"],
            "alveolar"             : ["t", "d", "s", "z", "n", "r", "4", "l"],
            "palatoalveolar"       : ["S", "Z", "tS", "dZ"],
            "palatal"              : ["J", "L", "j", "j~", "c", "J\\"],
            "velar"                : ["k", "g", "x", "G", "R\\"],
            "pause"                : ["sil", "<p:>"]
        }

    def get_list_phonological(self):
        return self.list_phonological

    def get_list_phonological_keys(self):
        keys=self.list_phonological.keys()
        return list(keys)

    def get_d1(self):
        keys=self.get_list_phonological_keys()
        dict_1={"xmin":[],"xmax":[],"phoneme":[],"phoneme_code":[]}
        for k in keys:
            dict_1[k]=[]
        return dict_1

    def get_d2(self):
        keys=self.get_list_phonological_keys()
        dict_2={"n_frame":[],"phoneme":[],"phoneme_code":[]}
        for k in keys:
            dict_2[k]=[]
        return dict_2

    def get_list_phonemes(self):
        keys=self.get_list_phonological_keys()
        phon=[]
        for k in keys:
            phon.append(self.list_phonological[k])
        phon=np.hstack(phon)

        return np.unique(phon)


def main():
    phon=Phonological()
    keys=phon.get_list_phonological_keys()
    print(keys)
    d1=phon.get_d1()
    print(d1)
    d2=phon.get_d2()
    print(d2)
    ph=phon.get_list_phonemes()
    print(ph)

if __name__=="__main__":
    main()


