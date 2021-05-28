#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
resume_epoch=

# feature configuration
do_delta=false

train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
skip_lm_training=true

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10
use_valbest_average=false
skip_confmod_training=false

# data
chime4_data=/export/corpora5/CHiME4/CHiME3 # JHU setup
wsj0=/export/corpora5/LDC/LDC93S6B            # JHU setup
wsj1=/export/corpora5/LDC/LDC94S13B           # JHU setup

## FDLP spectrum parameters ##

nfilters=80
order=150
fduration=1.5
frate=100
overlap_fraction=0.25
coeff_num=100
lp=1; hp=100
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

append=""
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
# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set_ori=tr05_multi_noisy_si284 # tr05_multi_noisy (original training data) or tr05_multi_noisy_si284 (add si284 data)
train_dev_ori=dt05_multi_isolated_1ch_track
recog_set_ori="\
et05_real_beamformit_5mics et05_real_beamformit_2mics et05_real_isolated_1ch_track \
dt05_real_isolated_1ch_track dt05_simu_isolated_1ch_track et05_simu_isolated_1ch_track \
et05_simu_beamformit_2mics \
et05_simu_beamformit_5mics \
"

train_set=${train_set_ori}_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
train_dev=${train_dev_ori}_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
train_set_temp1=tr05_real_noisy_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
train_set_temp2=tr05_simu_noisy_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
train_set_temp3=train_si284_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
temp=tr05_multi_noisy_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
temp_dev1=dt05_simu_isolated_1ch_track_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
temp_dev2=dt05_real_isolated_1ch_track_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}

[ ! -d data/$train_set_temp1 ] && ./utils/copy_data_dir.sh data/tr05_real_noisy data/$train_set_temp1
[ ! -d data/$train_set_temp2 ] && ./utils/copy_data_dir.sh data/tr05_simu_noisy data/$train_set_temp2
[ ! -d data/$train_set_temp3 ] && ./utils/copy_data_dir.sh data/train_si284 data/$train_set_temp3
recog_set=""
for s in $recog_set_ori; do
  s_append=${s}_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
  recog_set="$recog_set ${s_append}"
  [ ! -d data/${s_append} ] && ./utils/copy_data_dir.sh data/$s data/${s_append}
done

dumpdir=dump_fbank_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
fbankdir=fbank_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}


dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
nlsyms=data/lang_1char/non_lang_syms.txt

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
#mkdir -p ${lmexpdir}


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ! -z $resume_epoch ]; then
  resume=$expdir/results/snapshot.ep.${resume_epoch}
  echo "Loading from snapshot: $resume"
fi


skip_lm_training=false
recog_set_reduced=`echo $recog_set | cut -d' ' -f1-3`

# All the models used for CL
model1="../wsj/exp/train_si284_nf80_ord150_fdur1.5_range0100_ola0.25_frate100_DEBUGGED_gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train_no_preprocess_FDLP/results/model.last10.avg.best"
model1_pm="../wsj/exp/train_si284_nf80_ord150_fdur1.5_range0100_ola0.25_frate100_DEBUGGED_gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train_no_preprocess_FDLP/confidence_model/WSJ_px_VAE_enc2l_dec2l_512nodes_bn50/exp_1.dir/exp_1__epoch_100.model"
model2="../reverb/exp/tr_simu_8ch_si284_nf80_ord150_fdur1.5_range1450_ola0.25_frate100__gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train/results/model.last10.avg.best"
model2_pm="../reverb/exp/tr_simu_8ch_si284_nf80_ord150_fdur1.5_range1450_ola0.25_frate100__gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train/confidence_model/WSJ_px_VAE_enc2l_dec2l_512nodes_bn50/exp_1.dir/exp_1__epoch_100.model"
model3="../chime4/exp/tr05_multi_noisy_si284_nf80_ord150_fdur1.5_range1100_ola0.25_frate100__gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train/results/model.last10.avg.best"
model3_pm="../chime4/exp/tr05_multi_noisy_si284_nf80_ord150_fdur1.5_range1100_ola0.25_frate100__gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train/confidence_model/WSJ_px_VAE_enc2l_dec2l_512nodes_bn50/exp_1.dir/exp_1__epoch_100.model"

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 5: Decoding"
    nj=32
    temperature=20
    pids=() # initialize pids
    for rtask in ${recog_set_reduced}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}_CL_VAE_bn50_temp${temperature}_3stream
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --temperature ${temperature} \
            --api "cl" \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model_list ${model1},${model2},${model3} \
            --pm_list ${model1_pm},${model2_pm},${model3_pm} \
            ${recog_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
