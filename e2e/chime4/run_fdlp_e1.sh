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

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make the following data preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    wsj0_data=${chime4_data}/data/WSJ0
    skip=false
    if $skip; then
    local/clean_wsj0_data_prep.sh ${wsj0_data}
    local/clean_chime4_format_data.sh
    echo "beamforming for multichannel cases"
    local/run_beamform_2ch_track.sh --cmd "${train_cmd}" --nj 20 \
        ${chime4_data}/data/audio/16kHz/isolated_2ch_track enhan/beamformit_2mics
    local/run_beamform_6ch_track.sh --cmd "${train_cmd}" --nj 20 \
        ${chime4_data}/data/audio/16kHz/isolated_6ch_track enhan/beamformit_5mics
    echo "prepartion for chime4 data"
    local/real_noisy_chime4_data_prep.sh ${chime4_data}
    local/simu_noisy_chime4_data_prep.sh ${chime4_data}
    echo "test data for 1ch track"
    local/real_enhan_chime4_data_prep.sh isolated_1ch_track ${chime4_data}/data/audio/16kHz/isolated_1ch_track
    local/simu_enhan_chime4_data_prep.sh isolated_1ch_track ${chime4_data}/data/audio/16kHz/isolated_1ch_track
    echo "test data for 2ch track"
    local/real_enhan_chime4_data_prep.sh beamformit_2mics ${PWD}/enhan/beamformit_2mics
    local/simu_enhan_chime4_data_prep.sh beamformit_2mics ${PWD}/enhan/beamformit_2mics
    echo "test data for 6ch track"
    local/real_enhan_chime4_data_prep.sh beamformit_5mics ${PWD}/enhan/beamformit_5mics
    local/simu_enhan_chime4_data_prep.sh beamformit_5mics ${PWD}/enhan/beamformit_5mics
  fi
    # Additionally use WSJ clean data. Otherwise the encoder decoder is not well trained
    local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/wsj_format_data.sh
fi

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
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    tasks="${recog_set} ${train_set_temp1} ${train_set_temp2} ${train_set_temp3}"
    for x in ${tasks}; do
        utils/copy_data_dir.sh data/${x} data-fbank/${x}
        local_pyspeech/make_FDLPspectrum_feats.sh --cmd "$train_cmd" --nj 40 \
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

    echo "combine real and simulation data"
    utils/combine_data.sh --extra_files utt2num_frames data-fbank/${temp} data-fbank/${train_set_temp1} data-fbank/${train_set_temp2}
    utils/combine_data.sh --extra_files utt2num_frames data-fbank/${train_set} data-fbank/${temp} data-fbank/${train_set_temp3}
    utils/combine_data.sh --extra_files utt2num_frames data-fbank/${train_dev} data-fbank/${temp_dev1} data-fbank/${temp_dev2}

    # compute global CMVN
    compute-cmvn-stats scp:data-fbank/${train_set}/feats.scp data-fbank/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16}/${USER}/espnet-data/egs/chime4/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16}/${USER}/espnet-data/egs/chime4/asr1/dump/${train_dev}/delta${do_delta}/storage \
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
#mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! ${skip_lm_training}; then
    echo "stage 3: LM Preparation"
    if [ ${use_wordlm} = true ]; then
        lmdatadir=data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}
        cut -f 2- -d" " data-fbank/${train_set}/text > ${lmdatadir}/train_trans.txt
        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
                | grep -v "<" | tr "[:lower:]" "[:upper:]" > ${lmdatadir}/train_others.txt
        cut -f 2- -d" " data-fbank/${train_dev}/text > ${lmdatadir}/valid.txt
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
        --patience 100 \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && ! ${skip_confmod_training}; then
    echo "stage 5: Confidence VAE model training"
    data_dump=true
    if $data_dump; then
      # compute global CMVN
      compute-cmvn-stats scp:data-fbank/${temp}/feats.scp data-fbank/${temp}/cmvn.ark
      feat_tr_dir_onlyreverb=${dumpdir}/${temp}/delta${do_delta}; mkdir -p ${feat_tr_dir_onlyreverb}

      # Dump only reverb data first
      if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir_onlyreverb}/storage ]; then
      utils/create_split_dir.pl \
          /export/b{10,11,12,13}/${USER}/espnet-data/egs/reverb/asr1/dump/${temp}/delta${do_delta}/storage \
          ${feat_tr_dir_onlyreverb}/storage
      fi
      dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
          data-fbank/${temp}/feats.scp data-fbank/${temp}/cmvn.ark exp/dump_feats/train ${feat_tr_dir_onlyreverb}
    fi
    local_pyspeech/train_VAE.sh \
      --stage 0 \
      --vae_type "normal" \
      --use_gpu  true \
      --skip_cmvn true \
      --out_dist "laplace" \
      --data_dir ${dumpdir} \
      --hybrid_dir ${expdir}/confidence_model \
      --feat_type FDLP_spectrum \
      --train_set ${temp}/delta${do_delta}\
      --dev_set ${train_dev}/delta${do_delta} \
      --nn_name WSJ_px_VAE_enc2l_dec2l_512nodes_bn50 \
      --encoder_num_layers 2 \
      --decoder_num_layers 2 \
      --weight_decay 0 \
      --num_egs_jobs 10 \
      --ali_type "ignore" \
      --hidden_dim 512 \
      --bn_dim 50 \
      --batch_size 64 \
      --epochs 100 || exit 1 ;
fi

skip_lm_training=false
recog_set_reduced=`echo $recog_set | cut -d' ' -f1-3`
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 5: Decoding"
    nj=32
    skip=false
    if $skip; then
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
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
    fi
    pids=() # initialize pids
    for rtask in ${recog_set_reduced}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        
        if $skip; then
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
        fi
        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
