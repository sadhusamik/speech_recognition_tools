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
coeff_num=450
lp=1; hp=450
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

train_set_ori=tr_simu_8ch_si284
train_dev_ori=dt_mult_1ch
#recog_set_ori="dt_real_8ch_beamformit dt_simu_8ch_beamformit et_real_8ch_beamformit et_simu_8ch_beamformit dt_real_1ch_wpe dt_simu_1ch_wpe et_real_1ch_wpe et_simu_1ch_wpe"
recog_set_ori="et_real_8ch_beamformit et_real_1ch et_real_1ch_wpe et_simu_1ch_wpe dt_real_1ch dt_simu_1ch"


if $no_norm; then
  train_set=${train_set_ori}_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_nonorm_${append}
  train_dev=${train_dev_ori}_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_nonorm_${append}
  train_set_temp1=tr_simu_8ch_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_nonorm_${append}
  train_set_temp2=train_si284_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_nonorm_${append}
  recog_set=""
  [ ! -d data/$train_set_temp1 ] && ./utils/copy_data_dir.sh data/tr_simu_8ch data/$train_set_temp1
  [ ! -d data/$train_set_temp2 ] && ./utils/copy_data_dir.sh data/train_si284 data/$train_set_temp2
  for s in $recog_set_ori; do
    s_append=${s}_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_nonorm_${append}
    recog_set="$recog_set ${s_append}"
    [ ! -d data/${s_append} ] && ./utils/copy_data_dir.sh data/$s data/${s_append}
  done
else
  train_set=${train_set_ori}_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
  train_dev=${train_dev_ori}_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
  train_set_temp1=tr_simu_8ch_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
  train_set_temp2=train_si284_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
  recog_set=""
  [ ! -d data/$train_set_temp1 ] && ./utils/copy_data_dir.sh data/tr_simu_8ch data/$train_set_temp1
  [ ! -d data/$train_set_temp2 ] && ./utils/copy_data_dir.sh data/train_si284 data/$train_set_temp2
  for s in $recog_set_ori; do
    s_append=${s}_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
    recog_set="$recog_set ${s_append}"
    [ ! -d data/${s_append} ] && ./utils/copy_data_dir.sh data/$s data/${s_append}
  done
fi

if $no_norm; then
    dumpdir=dump_fbank_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_nonorm_${append}
    fbankdir=fbank_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_nonorm_${append}
  else
    dumpdir=dump_fbank_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
    fbankdir=fbank_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append}
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
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

skip_lm_training=false
recog_set_reduced=`echo $recog_set | cut -d' ' -f1-2`

# All the models used for CL
model1="../wsj/exp/train_si284_nf80_ord150_fdur1.5_range0100_ola0.25_frate100_DEBUGGED_gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train_no_preprocess_FDLP/split2/1/results/model.last10.avg.best"
model1_pm="../wsj/exp/train_si284_nf80_ord150_fdur1.5_range0100_ola0.25_frate100_DEBUGGED_gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train_no_preprocess_FDLP/split2/1/confidence_model/WSJ_px_VAE_enc2l_dec2l_512nodes_bn50/exp_1.dir/exp_1__epoch_100.model"
model2="../wsj/exp/train_si284_nf80_ord150_fdur1.5_range0100_ola0.25_frate100_DEBUGGED_gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train_no_preprocess_FDLP/split2/2/results/model.last10.avg.best"
model2_pm="../wsj/exp/train_si284_nf80_ord150_fdur1.5_range0100_ola0.25_frate100_DEBUGGED_gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train_no_preprocess_FDLP/split2/2/confidence_model/WSJ_px_VAE_enc2l_dec2l_512nodes_bn50/exp_1.dir/exp_1__epoch_100.model"

model3="../reverb/exp/tr_simu_8ch_si284_nf80_ord150_fdur1.5_range1450_ola0.25_frate100__gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train/split2/1/results/model.last10.avg.best"
model3_pm="../reverb/exp/tr_simu_8ch_si284_nf80_ord150_fdur1.5_range1450_ola0.25_frate100__gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train/split2/1/confidence_model/WSJ_px_VAE_enc2l_dec2l_512nodes_bn50/exp_1.dir/exp_1__epoch_100.model"
model4="../reverb/exp/tr_simu_8ch_si284_nf80_ord150_fdur1.5_range1450_ola0.25_frate100__gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train/split2/2/results/model.last10.avg.best"
model4_pm="../reverb/exp/tr_simu_8ch_si284_nf80_ord150_fdur1.5_range1450_ola0.25_frate100__gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train/split2/2/confidence_model/WSJ_px_VAE_enc2l_dec2l_512nodes_bn50/exp_1.dir/exp_1__epoch_100.model"


model5="../chime4/exp/tr05_multi_noisy_si284_nf80_ord150_fdur1.5_range1100_ola0.25_frate100__gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train/split2/1/results/model.last10.avg.best"
model5_pm="../chime4/exp/tr05_multi_noisy_si284_nf80_ord150_fdur1.5_range1100_ola0.25_frate100__gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train/split2/1/confidence_model/WSJ_px_VAE_enc2l_dec2l_512nodes_bn50/exp_1.dir/exp_1__epoch_100.model"
model6="../chime4/exp/tr05_multi_noisy_si284_nf80_ord150_fdur1.5_range1100_ola0.25_frate100__gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train/split2/2/results/model.last10.avg.best"
model6_pm="../chime4/exp/tr05_multi_noisy_si284_nf80_ord150_fdur1.5_range1100_ola0.25_frate100__gw_none_cochlear_omw1_alp1_fx_1_bet2.5_wf1_pytorch_train/split2/2/confidence_model/WSJ_px_VAE_enc2l_dec2l_512nodes_bn50/exp_1.dir/exp_1__epoch_100.model"

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32
    temperature=1
    score_type=prod
    rule=sum
    pids=() # initialize pids
    for rtask in ${recog_set_reduced}; do
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

        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}_CL_VAE_bn50_temp${temperature}_6stream_splitdata_ST_${score_type}_RULE_${rule}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        mkdir -p ${expdir}/${decode_dir}/pm_scores/
        mkdir -p ${expdir}/${decode_dir}/post_dump/

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --temperature ${temperature} \
            --score_type ${score_type} \
            --rule ${rule} \
            --pm_save_path ${expdir}/${decode_dir}/pm_scores/score.JOB \
            --posterior_save_path ${expdir}/${decode_dir}/post_dump/post.JOB \
            --api "cl" \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model_list ${model1},${model3},${model2},${model4},${model5},${model6}  \
            --pm_list ${model1_pm},${model3_pm},${model2_pm},${model4_pm},${model5_pm},${model6_pm} \
            ${recog_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
