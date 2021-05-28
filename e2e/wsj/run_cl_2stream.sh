#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=1        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump_fdlp   # directory to dump full features
fbankdir=fbank_fdlp
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1
resume_epoch=""

# feature configuration
do_delta=false
no_norm=false # This option does not change anything

# sample filtering
min_io_delta=4  # samples with `len(input) - len(output) * min_io_ratio < min_io_delta` will be removed.

# config files
preprocess_config=conf/no_preprocess.yaml  # use conf/specaug.yaml for data augmentation
train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
skip_lm_training=false  # for only using end-to-end ASR model without LM
use_wordlm=true         # false means to train/use a character LM
lm_vocabsize=65000      # effective only for word LMs
lm_resume=              # specify a snapshot file to resume LM training
lmtag=                  # tag for managing LMs

# decoding parameter
recog_model=model.acc.best   # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=10                 # the number of ASR models to be averaged
use_valbest_average=false    # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.

# Confidence Model related
skip_confmod_training=false

## FDLP spectrum parameters ##

nfilters=80
order=150
fduration=1.5
frate=100
overlap_fraction=0.25
coeff_num=100
lp=0; hp=100
coeff_range="$lp,$hp"
wf=1
# FILTER CONFIGURATION
om_w=1
alp=1
fixed=1
bet=2.5
fb="cochlear"
# FOR MODULATION GAMMA WEIGHT
gw="no"
scale=20
shape=1.5
pk=3

append="DEBUGGED"
if [ $gw == "yes" ] ;then
  gamma_weight="$scale,$shape,$pk"
  append="${append}_gw_sc_${scale}_sh_${shape}_pk_${pk}"
else
  gamma_weight="None"
  append="${append}_gw_none"
fi

if [ $fb == "cochlear" ] ;then
  fbank_type="cochlear,$om_w,$alp,$fixed,$bet,$wf"
  append="${append}_cochlear_omw${om_w}_alp${alp}_fx_${fixed}_bet${bet}_wf${wf}"
elif [ $fb == "mel" ] ;then
  fbank_type="mel,$wf"
  append="${append}_mel_wf${wf}"
else
  echo "Incorrect filter bank type $fb, use mel or cochlear"
  exit 1
fi

add_lindist_test_data=false
noises="babble street"
snrs="20 40"
reverbs=

# data
wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set_ori=train_si284
train_dev_ori=test_dev93
train_test_ori=test_eval92
recog_set_ori="test_dev93 test_eval92"


if $no_norm; then
  train_set=train_si284_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_nonorm_${append}
  train_dev=test_dev93_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_nonorm_${append}
  train_test=test_eval92_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_nonorm_${append}
else
  train_set=train_si284_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
  train_dev=test_dev93_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
  train_test=test_eval92_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
fi

if $no_norm; then
  dumpdir=dump_fdlp_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_nonorm_${append}
  fbankdir=fbank_pyspeech_fdlp_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_nonorm_${append}
else
  dumpdir=dump_fdlp_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
  fbankdir=fbank_pyspeech_fdlp_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
fi

[ ! -d data/$train_set ] && ./utils/copy_data_dir.sh data/train_si284 data/$train_set
[ ! -d data/$train_dev ] && ./utils/copy_data_dir.sh data/test_dev93 data/$train_dev
[ ! -d data/$train_test ] && ./utils/copy_data_dir.sh data/test_eval92 data/$train_test

recog_set="$train_test"

for reverb  in $reverbs; do
   if $no_norm; then
    tset=test_eval92_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${reverb}_nonorm_${append}
    recog_set="$recog_set $tset"
  else
    tset=test_eval92_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${reverb}_${append}
    recog_set="$recog_set $tset"
  fi
done

for noise in $noises; do
  for snr in $snrs; do
   if $no_norm; then
    tset=test_eval92_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${noise}_${snr}_nonorm_${append}
    recog_set="$recog_set $tset"
  else
    tset=test_eval92_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${noise}_${snr}_${append}
    recog_set="$recog_set $tset"
  fi
  done
done

if $add_lindist_test_data; then
  if $no_norm; then
    tset=test_eval92_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_diff_nonorm_${append}
  else
    tset=test_eval92_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_diff_${append}
  fi
  recog_set="$recog_set $tset"
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt


if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}_FDLP
mkdir -p ${expdir}

recog_set="$train_test"
skip_lm_training=false

# All the models used for CL
model1="exp/train_si284_nf80_ord150_fdur1.5_range0100_ola0.25_frate100_DEBUGGED_gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train_no_preprocess_FDLP/results/model.last10.avg.best"
model1_pm="exp/train_si284_nf80_ord150_fdur1.5_range0100_ola0.25_frate100_DEBUGGED_gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train_no_preprocess_FDLP/confidence_model/WSJ_px_VAE_enc2l_dec2l_300nodes/exp_1.dir/exp_1__epoch_100.model"
model2="../reverb/exp/tr_simu_8ch_si284_nf80_ord150_fdur1.5_range1450_ola0.25_frate100__gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train/results/model.last10.avg.best"
model2_pm="../reverb/exp/tr_simu_8ch_si284_nf80_ord150_fdur1.5_range1450_ola0.25_frate100__gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train/confidence_model/WSJ_px_VAE_enc2l_dec2l_300nodes/exp_1.dir/exp_1__epoch_70.model"

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        recog_opts=
        if ${skip_lm_training}; then
            if [ -z ${lmtag} ]; then
                lmtag="nolm"
            fi
        else
            if [ ${use_wordlm} = true ]; then
                recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
            else
                recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
            fi
        fi

        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}_CL
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --api "cl" \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model_list ${model1},${model2}  \
            --pm_list ${model1_pm},${model2_pm} \
            ${recog_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
