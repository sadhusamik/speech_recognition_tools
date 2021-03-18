export PYSPEECH_ROOT=$PWD/../../
export PYSPEECH_TOOLS=$PYSPEECH_ROOT/src
export PATH=$PATH:$PYSPEECH_TOOLS
export KALDI_ROOT=$PYSPEECH_ROOT/tools/kaldi

#Add pyspeech stuff 
export PATH=$PYSPEECH_TOOLS/hmm/:$PYSPEECH_TOOLS/gmm/:$PYSPEECH_TOOLS/utils:$PYSPEECH_TOOLS/utils_pytorch_kaldi:$PYSPEECH_TOOLS/featgen/:$PYSPEECH_TOOLS/nnet/:$PYSPEECH_TOOLS/rbm:$PYSPEECH_ROOT/tools/pytorch-kaldi:$PATH
PYTHONPATH=$PYSPEECH_TOOLS/featgen

MAIN_ROOT=$PYSPEECH_ROOT/tools/espnet

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build
. "${MAIN_ROOT}"/tools/activate_python.sh && . "${MAIN_ROOT}"/tools/extra_path.sh
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export PATH=$PYSPEECH_TOOLS/hmm/:$PYSPEECH_TOOLS/gmm/:$PYSPEECH_TOOLS/utils:$PYSPEECH_TOOLS/utils_pytorch_kaldi:$PYSPEECH_TOOLS/featgen/:$PYSPEECH_TOOLS/nnet/:$PYSPEECH_TOOLS/rbm:$PYSPEECH_ROOT/tools/pytorch-kaldi:$PATH
export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
