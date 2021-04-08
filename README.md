## Speech Recognition Tools

This repository contains miscellaneous tools for Automatic Speech Recognition(ASR) based on Kaldi or ESPNet

#### To use ESPnet recipes
* Install ESPnet (https://github.com/espnet/espnet) under `speech_recognition_tools/tools/`.
* If you already have ESPnet installed you can provide a symbolic link to it
  
#### To use Kaldi recipes
* Install my Kaldi fork (https://github.com/sadhusamik/kaldi) under `speech_recognition_tools/tools/`
* You can use any newer versions of Kaldi but might have to change some scripts
* If you already have Kaldi installed you can provide a symbolic link to it

### FDLP-spectrogram (with ESPnet)
* Paper: (https://arxiv.org/abs/2103.14129)
* The WSJ and REVERB recipes for FDLP-spectrogram can be found under `e2e/wsj` and `e2e/reverb`. 
* The script `run_fdlp_e1.sh` runs the standard ESPnet recipes for each dataset with FDLP-spectrogram features instead of the usual mel-spectrogram (log filter-bank energy)
* The script `run_melspec.sh` runs the ESPnet baseline mel-spectrogram features. NOTE: This is different from the Kaldi recipe of mel-spectrogram 
* The script `run_fdlp_e1.sh` provides the template to use FDLP-spectrogram for other datasets 


| Data set                                                        |  mel-spectrogram  |  FDLP-spectrogram |
|-----------------------------------------------------------------|:-----------------:|:-----------------:|
| WSJ (test_eval92)                                               |        5.1        |        4.8        |
| REVERB (et_real_1ch / et_real_1ch_wpe / et_real_8ch_beamformit) | 23.2 / 20.7 / 9.2 | 19.4 / 18.0 / 7.2 |
| CHIME4 (et05_real_isolated_1ch_track / et05_real_beamformit_2mics / et05_real_beamformit_5mics) | 23.7 / 20.4 / 16.8 | 23.4 / 19.5 / 15.8 |