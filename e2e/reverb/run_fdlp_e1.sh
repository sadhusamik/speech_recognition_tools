#!/bin/bash

# Copyright 2018 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump_fdlp   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1
resume_epoch=""

# feature configuration
do_delta=false
no_norm=false #This option does not make any changes

# config files
preprocess_config=conf/no_preprocess.yaml
train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs
skip_lm_training=false

# model average realted (only for transformer)
n_average=10                 # the number of ASR models to be averaged
use_valbest_average=false    # if true, the validation `n_average`-best ASR models will be averaged.

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# Dereverberation Measures
compute_se=true # flag for turing on computation of dereverberation measures
enable_pesq=false # please make sure that you or your institution have the license to report PESQ before turning on this flag
nch_se=8

# data
reverb=/export/corpora5/REVERB_2014/REVERB    # JHU setup
wsjcam0=/export/corpora3/LDC/LDC95S24/wsjcam0 # JHU setup
wsj0=/export/corpora5/LDC/LDC93S6B            # JHU setup
wsj1=/export/corpora5/LDC/LDC94S13B           # JHU setup
wavdir=${PWD}/wav # place to store WAV files


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

train_set_ori=tr_simu_8ch_si284
train_dev_ori=dt_mult_1ch
#recog_set_ori="dt_real_8ch_beamformit dt_simu_8ch_beamformit et_real_8ch_beamformit et_simu_8ch_beamformit dt_real_1ch_wpe dt_simu_1ch_wpe et_real_1ch_wpe et_simu_1ch_wpe"
recog_set_ori="et_real_8ch_beamformit et_real_1ch et_real_1ch_wpe et_simu_1ch_wpe dt_real_1ch dt_simu_1ch"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make the following data preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    wavdir=${PWD}/wav # set the directory of the multi-condition training WAV files to be generated
    echo "stage 0: Data preparation"
    local/generate_data.sh --wavdir ${wavdir} ${wsjcam0}
    local/prepare_simu_data.sh --wavdir ${wavdir} ${reverb} ${wsjcam0}
    local/prepare_real_data.sh --wavdir ${wavdir} ${reverb}

    # Run WPE and Beamformit
    local/run_wpe.sh
    local/run_beamform.sh ${wavdir}/WPE/
    if ${compute_se}; then
      if [ ! -d local/REVERB_scores_source ] || [ ! -d local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/evaltools/SRMRToolbox ] || [ ! -f local/PESQ ]; then
        # download and install speech enhancement evaluation tools
        local/download_se_eval_tool.sh
      fi
      pesqdir=${PWD}/local
      local/compute_se_scores.sh --nch ${nch_se} --enable_pesq ${enable_pesq} ${reverb} ${wavdir} ${pesqdir}
      cat exp/compute_se_${nch_se}ch/scores/score_SimData
      cat exp/compute_se_${nch_se}ch/scores/score_RealData
    fi

    # Additionally use WSJ clean data. Otherwise the encoder decoder is not well trained
    local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/wsj_format_data.sh
fi

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
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    tasks="${recog_set} ${train_set_temp1} ${train_set_temp2}"
    for x in ${tasks}; do
        utils/copy_data_dir.sh data/${x} data-fbank/${x}
        local_pyspeech/make_FDLPspectrum_feats.sh --cmd "$train_cmd" --nj 20 \
          --nfilters $nfilters \
          --order $order \
          --fduration $fduration \
          --frate $frate \
          --coeff_range $coeff_range \
          --coeff_num $coeff_num \
          --overlap_fraction $overlap_fraction \
          --fbank_type $fbank_type \
          --gamma_weight $gamma_weight \
          --write_utt2num_frames true \
          data-fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data-fbank/${x}
    done
   
    echo "combine reverb simulation and wsj clean training data"
    utils/combine_data.sh data-fbank/${train_set} data-fbank/${train_set_temp1} data-fbank/${train_set_temp2}
    echo "combine real and simulation development data"
    
    if $no_norm; then
        utils/combine_data.sh data-fbank/${train_dev} \
          data-fbank/dt_real_1ch_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_nonorm_${append} \
          data-fbank/dt_simu_1ch_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_nonorm_${append} 
      else
        utils/combine_data.sh data-fbank/${train_dev} \
          data-fbank/dt_real_1ch_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append} \
          data-fbank/dt_simu_1ch_nf${nfilters}_ord${order}_fdur${fduration}_range${lp}${hp}_ola${overlap_fraction}_frate${frate}_${append} 
    fi
    # compute global CMVN
    compute-cmvn-stats scp:data-fbank/${train_set}/feats.scp data-fbank/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/reverb/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/reverb/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data-fbank/${train_set}/feats.scp data-fbank/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
        data-fbank/${train_dev}/feats.scp data-fbank/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
            data-fbank/${rtask}/feats.scp data-fbank/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
nlsyms=data/lang_1char/non_lang_syms.txt
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data-fbank/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data-fbank/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data-fbank/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data-fbank/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

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
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! ${skip_lm_training}; then
    echo "stage 3: LM Preparation"
    if [ ${use_wordlm} = true ]; then
        lmdatadir=data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}
        cut -f 2- -d" " data/${train_set}/text > ${lmdatadir}/train_trans.txt
        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
                | grep -v "<" | tr "[:lower:]" "[:upper:]" > ${lmdatadir}/train_others.txt
        cut -f 2- -d" " data/${train_dev}/text > ${lmdatadir}/valid.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
        lmdatadir=data/local/lm_train
        lmdict=${dict}
        mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text \
            | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt
        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
            | grep -v "<" | tr "[:lower:]" "[:upper:]" \
            | text2token.py -n 1 | cut -f 2- -d" " > ${lmdatadir}/train_others.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
    fi
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${lmdict}
fi

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

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --patience 50 \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi

skip_lm_training=false
recog_set_reduced=`echo $recog_set | cut -d' ' -f1-4`
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
       [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
       [[ $(get_yaml.py ${train_config} model-module) = *maskctc* ]] || \
       [[ $(get_yaml.py ${train_config} etype) = transformer ]] || \
       [[ $(get_yaml.py ${train_config} dtype) = transformer ]]; then
        average_opts=
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            average_opts="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
        fi
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${recog_model} \
                               --num ${n_average} \
                               ${average_opts}
    fi
    
    pids=() # initialize pids
    for rtask in ${recog_set_reduced}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        if [ ${use_wordlm} = true ]; then
            decode_dir=${decode_dir}_wordrnnlm_${lmtag}
        else
            decode_dir=${decode_dir}_rnnlm_${lmtag}
        fi
        
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
          feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        fi
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            ${recog_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    echo "Report the result"
    decode_part_dir=$(basename ${decode_config%.*})
    if [ ${use_wordlm} = true ]; then
	decode_part_dir=${decode_part_dir}_wordrnnlm_${lmtag}
    else
	decode_part_dir=${decode_part_dir}_rnnlm_${lmtag}
    fi
    local/get_results.sh ${nlsyms} ${dict} ${expdir} ${decode_part_dir}
    echo "Finished"
fi
