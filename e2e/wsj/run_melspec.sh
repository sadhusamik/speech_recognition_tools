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
dumpdir=dump_fbank   # directory to dump full features
fbankdir=fbank
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

# Feature Configuration

nfilters=80
nfft=2048
frate=100

wf=1
# FILTER CONFIGURATION
om_w=1
alp=1
fixed=1
bet=2.5
fb="mel"

append=""

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


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/wsj_format_data.sh
fi


if $no_norm; then
  train_set=train_si284_nf${nfilters}_nfft${nfft}_frate${frate}_nonorm_${append}
  train_dev=test_dev93_nf${nfilters}_nfft${nfft}_frate${frate}_nonorm_${append}
  train_test=test_eval92_nf${nfilters}_nfft${nfft}_frate${frate}_nonorm_${append}
else
  train_set=train_si284_nf${nfilters}_nfft${nfft}_frate${frate}_${append}
  train_dev=test_dev93_nf${nfilters}_nfft${nfft}_frate${frate}_${append}
  train_test=test_eval92_nf${nfilters}_nfft${nfft}_frate${frate}_${append}
fi

if $no_norm; then
    dumpdir=dump_fbank_${nfilters}_${nfft}_frate${frate}_nonorm_${append}
    fbankdir=fbank_${nfilters}_${nfft}_frate${frate}_nonorm_${append}
  else
    dumpdir=dump_fbank_${nfilters}_${nfft}_frate${frate}_${append}
    fbankdir=fbank_${nfilters}_${nfft}_frate${frate}_${append}
fi

[ ! -d data/$train_set ] && ./utils/copy_data_dir.sh data/train_si284 data/$train_set
[ ! -d data/$train_dev ] && ./utils/copy_data_dir.sh data/test_dev93 data/$train_dev
[ ! -d data/$train_test ] && ./utils/copy_data_dir.sh data/test_eval92 data/$train_test

recog_set="$train_test"

for reverb in $reverbs; do 
   if $no_norm; then 
    tset=test_eval92_nf${nfilters}_nfft${nfft}_frate${frate}_${reverb}_nonorm_${append}
    recog_set="$recog_set $tset"
  else
    tset=test_eval92_nf${nfilters}_nfft${nfft}_frate${frate}_${reverb}_${append}
    recog_set="$recog_set $tset"
  fi
done

for noise in $noises; do
  for snr in $snrs; do
   if $no_norm; then 
      tset=test_eval92_nf${nfilters}_nfft${nfft}_frate${frate}_${noise}_${snr}_nonorm_${append}
      recog_set="$recog_set $tset"
    else
      tset=test_eval92_nf${nfilters}_nfft${nfft}_frate${frate}_${noise}_${snr}_${append}
      recog_set="$recog_set $tset"
    fi 
  done
done

if $add_lindist_test_data; then 
  if $no_norm; then
    tset=test_eval92_nf${nfilters}_nfft${nfft}_frate${frate}_diff_nonorm_${append}
  else
    tset=test_eval92_nf${nfilters}_nfft${nfft}_frate${frate}_diff_${append}
  fi
  recog_set="$recog_set $tset"
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases

    echo "stage 1: Feature Generation"
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in $train_set $train_dev $train_test; do
        local_pyspeech/make_melspectrum_feats.sh --cmd "$train_cmd" --nj 20 \
          --nfft $nfft \
          --nfilters $nfilters \
          --frate $frate \
          --fbank_type $fbank_type \
          --write_utt2num_frames true \
            data/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    for noise in $noises; do
      for snr in $snrs; do
       if $no_norm; then 
          tset=test_eval92_nf${nfilters}_nfft${nfft}_frate${frate}_${noise}_${snr}_nonorm_${append}
       else
          tset=test_eval92_nf${nfilters}_nfft${nfft}_frate${frate}_${noise}_${snr}_${append}
       fi
        [ ! -d data/$tset ] && ./utils/copy_data_dir.sh data/test_eval92 data/$tset
        
        local_pyspeech/make_melspectrum_feats.sh --cmd "$train_cmd" --nj 20 \
          --nfft $nfft \
          --nfilters $nfilters \
          --frate $frate \
          --add_noise ${noise},${snr} \
          --fbank_type $fbank_type \
          --write_utt2num_frames true \
            data/${tset} ${fbankdir}
        utils/fix_data_dir.sh data/${tset}
      done
    done

    for reverb in $reverbs; do 
       if $no_norm; then 
        tset=test_eval92_nf${nfilters}_nfft${nfft}_frate${frate}_${reverb}_nonorm_${append}
      else
        tset=test_eval92_nf${nfilters}_nfft${nfft}_frate${frate}_${reverb}_${append}
      fi
      [ ! -d data/$tset ] && ./utils/copy_data_dir.sh data/test_eval92 data/$tset
      local_pyspeech/make_melspectrum_feats.sh --cmd "$train_cmd" --nj 20 \
        --nfft $nfft \
        --nfilters $nfilters \
        --add_reverb $reverb \
        --frate $frate \
        --fbank_type $fbank_type \
        --write_utt2num_frames true \
          data/${tset} ${fbankdir}
      utils/fix_data_dir.sh data/${tset}
    done

    if $add_lindist_test_data; then 
      tset=test_eval92_nf${nfilters}_nfft${nfft}_frate${frate}_diff_${append}
      [ ! -d data/$tset ] && ./utils/copy_data_dir.sh data/test_eval92 data/$tset
      local_pyspeech/make_melspectrum_feats.sh --cmd "$train_cmd" --nj 20 \
        --nfft $nfft \
        --nfilters $nfilters \
        --frate $frate \
        --add_noise diff \
        --fbank_type $fbank_type \
        --write_utt2num_frames true \
          data/${tset} ${fbankdir}
      utils/fix_data_dir.sh data/${tset}
    fi
    
    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/wsj/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/wsj/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} --no_norm ${no_norm}\
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

     ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done

    ### Filter out short samples which lead to `loss_ctc=inf` during training
    ###  with the specified configuration.
    # Samples satisfying `len(input) - len(output) * min_io_ratio < min_io_delta` will be pruned.
    local/filtering_samples.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --data-json ${feat_tr_dir}/data.json \
        --mode-subsample "asr" \
        ${min_io_delta:+--min-io-delta $min_io_delta} \
        --output-json-path ${feat_tr_dir}/data.json
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this by setting skip_lm_training=true


if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! ${skip_lm_training}; then
    echo "stage 3: LM Preparation"

    if [ -z ${lmtag} ]; then
        lmtag=$(basename ${lm_config%.*})
        if [ ${use_wordlm} = true ]; then
            lmtag=${lmtag}_word${lm_vocabsize}
        fi
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}
    lmexpdir=exp/${lmexpname}
    mkdir -p ${lmexpdir}

    if [ ${use_wordlm} = true ]; then
        lmdatadir=data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}
        cut -f 2- -d" " data/${train_set}/text > ${lmdatadir}/train_trans.txt
        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
                | grep -v "<" | tr "[:lower:]" "[:upper:]" > ${lmdatadir}/train_others.txt
        cut -f 2- -d" " data/${train_dev}/text > ${lmdatadir}/valid.txt
        cut -f 2- -d" " data/${train_test}/text > ${lmdatadir}/test.txt
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
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_test}/text \
                | cut -f 2- -d" " > ${lmdatadir}/test.txt
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
        --test-label ${lmdatadir}/test.txt \
        --resume ${lm_resume} \
        --dict ${lmdict}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})
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

        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
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
    echo "Finished"
fi
