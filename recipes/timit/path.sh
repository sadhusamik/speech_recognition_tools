export PYSPEECH_ROOT=$PWD/../../
export PYSPEECH_TOOLS=$PYSPEECH_ROOT/src
export PATH=$PATH:$PYSPEECH_TOOLS
export KALDI_ROOT=$PYSPEECH_ROOT/tools/kaldi

#[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C


export PATH=$PYSPEECH_TOOLS/hmm/:$PYSPEECH_TOOLS/gmm/:$PYSPEECH_TOOLS/utils:$PYSPEECH_TOOLS/utils_pytorch_kaldi:$PYSPEECH_TOOLS/featgen/:$PYSPEECH_TOOLS/nnet/:$PYSPEECH_TOOLS/rbm:$PYSPEECH_ROOT/tools/pytorch-kaldi:$PATH

#source activate /export/b18/ssadhu/tools/python/envs/env_3.6/envs/env_2.7

#source activate /export/b18/ssadhu/tools/python/envs/env_3.6

# This is valid only for the CLSP grid at JHU, change to your own proper conda
# environment for running pytorch
source activate /export/b15/ssadhu/speech_recognition_tools/environment

#PYTHONPATH=$PYTHONPATH:$PYSPEECH_TOOLS/hmm
#PYTHONPATH=$PYTHONPATH:$PYSPEECH_TOOLS/gmm
#PYTHONPATH=$PYTHONPATH:$PYSPEECH_TOOLS/utils
#PYTHONPATH=$PYTHONPATH:$PYSPEECH_TOOLS/utils_pytorch_kaldi
PYTHONPATH=$PYTHONPATH:$PYSPEECH_TOOLS/featgen
PYTHONPATH=$PYTHONPATH:$PYSPEECH_ROOT/tools/Autoregressive-Predictive-Coding
#PYTHONPATH=$PYTHONPATH:$PYSPEECH_TOOLS/nnet
#PYTHONPATH=$PYTHONPATH:$PYSPEECH_TOOLS/rbm
#PYTHONPATH=$PYTHONPATH:$PYSPEECH_ROOT/tools/pytorch-kaldi
PYTHONPATH=$PYTHONPATH:$PYSPEECH_ROOT/tools/kaldi-io-for-python
#PYTHONPATH=$PYTHONPATH:$PYSPEECH_ROOT/tools/gmm-torch

export PYTHONPATH=$PYTHONPATH

CUDA_VISIBLE_DEVICES=`free-gpu`
