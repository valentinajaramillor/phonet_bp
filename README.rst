======================
Phonet - Brazilian Portuguese Adaptation
======================

**This is an adaptation of the original Phonet toolkit for Brazilian Portuguese phonological analysis.**

**Original repository:** https://github.com/jcvasquezc/phonet/tree/master  
**Original author:** J. C. Vásquez-Correa

This toolkit computes posterior probabilities of phonological classes from audio files for several groups of phonemes according to the mode and manner of articulation, adapted specifically for Brazilian Portuguese.


The list of the phonological classes available and the phonemes that are activated for each phonological class are observed in the following Table


========================    ================================================================================
Phonological class          Phonemes
========================    ================================================================================
High                        /i/, /u/, /ɨ/, /ĩ/, /ũ/
Upper Mid                   /e/, /o/, /ẽ/, /õ/
Lower Mid                   /ɛ/, /ɔ/
Low                         /a/, /ɐ/, /ɐ̃/
Front                       /i/, /e/, /ɛ/, /ĩ/, /ẽ/, /j/, /j̃/
Central                     /a/, /ɐ/, /ɨ/, /ɐ̃/
Back                        /u/, /o/, /ɔ/, /ũ/, /õ/, /w/, /w̃/
Stop Voiceless              /p/, /t/, /k/, /c/
Stop Voiced                 /b/, /d/, /ɡ/, /ɟ/
Fricative Voiceless         /f/, /s/, /ʃ/, /x/
Fricative Voiced            /v/, /z/, /ʒ/, /ð/, /β/, /ɣ/
Nasal                       /m/, /n/, /ɲ/
Trill                       /ʀ/
Tap                         /ɾ/
Lateral                     /l/, /ʎ/
Bilabial                    /p/, /b/, /m/, /β/, /w/, /w̃/
Labiodental                 /f/, /v/
Alveolar                    /t/, /d/, /s/, /z/, /n/, /r/, /ɾ/, /l/
Palatoalveolar              /ʃ/, /ʒ/, /tʃ/, /dʒ/
Palatal                     /ɲ/, /ʎ/, /j/, /j̃/, /c/, /ɟ/
Velar                       /k/, /ɡ/, /x/, /ɣ/, /ʀ/
Pause                       /sil/
========================    ================================================================================

Installation
============


From this repository::

    git clone https://github.com/valentinajaramillor/phonet_bp
    cd phonet
    python setup.py

Usage
=====

Supported features:

- Estimate probabilities of phonological classes for audio files

Script is called as follows

    python phonet_posteriors.py <file_or_folder_audio> <file_features> <static (true or false)> <plots (true or false)> <format (csv, txt, npy, torch)>

`Code <phonet/phonet_posteriors.py>`_


Training
====================================

If you want to train Phonet in your own language, or specific phonological classes that are not defined here, please refer to the folder `train <https://github.com/valentinajaramillor/phonet_bp/tree/master/phonet/train>`_ and follow the instructions there.



Reference
==================================

This work is based on the original Phonet toolkit by J. C. Vásquez-Correa.

If you use Phonet, please cite the following paper.

@inproceedings{Vasquez-Correa2019,
  author={J. C. Vásquez-Correa and P. Klumpp and J. R. Orozco-Arroyave and E. N\"oth},
  title={{Phonet: A Tool Based on Gated Recurrent Neural Networks to Extract Phonological Posteriors from Speech}},
  year=2019,
  booktitle={Proc. Interspeech 2019},
  pages={549--553},
  doi={10.21437/Interspeech.2019-1405},
  url={http://dx.doi.org/10.21437/Interspeech.2019-1405}
}
