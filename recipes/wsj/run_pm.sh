#!/bin/bash

## Train APC PM on posteriors on WSJ
# Samik Sadhu 

stage=0

exp_dir=exp
data_dir=data

train=true
decode=true
align=true
simplify_dict=false

. ./cmd.sh

. utils/parse_options.sh

wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B

. ./path.sh

mfcc=mfcc_clean
nj=50

if [ $stage -le 0 ]; then

  echo "## STAGE: Data Prepartion ##"
  
  local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;

  local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;

  utils/prepare_lang.sh  --position_dependent_phones false \
    data/local/dict_nosp \
    "<SPOKEN_NOISE>" \
    data/local/lang_tmp_nosp \
    data/lang_nosp || exit 1;

  local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;

  cp -r data $data_dir
fi

for x in silence_phones.txt nonsilence_phones.txt; do 
  cat $data_dir/data/local/dict_nosp/$x
done > conf/phones.txt

if [ $stage -le 1 ]; then
  
  echo "## STAGE: MFCC feature extraction ##"

  add_opts= #'--add_reverb=large_room' 
  for x in test_eval92 test_dev93 train_si284; do
    local_pyspeech/make_mfcc_feats.sh --nj $nj \
      --nfilters 23 \
      data/$x data/$x/$mfcc $add_opts || exit 1;
    local_pyspeech/get_cmvn.sh \
      data/$x data/$x/$mfcc/cmvn || exit 1;

  done

fi

if [ $stage -le 2 ]; then
  
  echo "## STAGE: Monophone Training ##"


  # Fetch the mfcc features for getting alignments
  for x in train_si284 test_dev93 test_eval92; do
    local_pyspeech/fetch_feats.sh $data_dir/$x $mfcc || exit 1;
  done

  if $train; then
    steps/train_mono.sh --boost-silence 1.25 --nj 80 --cmd "$train_cmd" \
      $data_dir/train_si284 \
      $data_dir/lang_nosp \
      $exp_dir/mono0a || exit 1;
  fi

  if $decode; then
    utils/mkgraph.sh $data_dir/lang_nosp_test_tgpr \
      $exp_dir/mono0a \
      $exp_dir/mono0a/graph_nosp_tgpr && \
      steps/decode.sh --nj 8 --cmd "$decode_cmd" $exp_dir/mono0a/graph_nosp_tgpr \
        $data_dir/test_eval92 $exp_dir/mono0a/decode_nosp_tgpr_eval92
  fi
fi
echo "DONE"
printf "\n"


if [ $stage -le 3 ]; then

 echo "## STAGE: Hybrid Training ##"

 
 if $train; then

  # Fetch the mfcc features for getting proper alignments
    for x in train_si284 test_dev93 test_eval92; do 
      local_pyspeech/fetch_feats.sh $data_dir/$x $mfcc || exit 1;
    done

  steps/align_si.sh --nj 80 --cmd "$train_cmd" \
    $data_dir/train_si284 \
    $data_dir/lang_nosp \
    $exp_dir/mono0a \
    $exp_dir/mono0a_ali || exit 1;

    # DNN hybrid system training parameters
    dnn_mem_reqs="--mem 1G"
    dnn_extra_opts="--num_epochs 5 --num-epochs-extra 2 --add-layers-period 1 --shrink-interval 3"

    # Run Hybrid training with modulation spectral features
    #-l 'hostname=hostname=b1[12345678]*|c*' --gpu 1"

    steps/nnet2/train_tanh_fast.sh \
      --mix-up 5000 \
      --initial-learning-rate 0.015 \
      --final-learning-rate 0.002 \
      --num-hidden-layers 5  \
      --num_threads 8 \
      --parallel_opts "--num-threads 8" \
      --hidden_layer_dim 512 \
      --num-jobs-nnet 16 \
      --splice_width 4 \
      --cmd "$train_cmd" \
      "${dnn_train_extra_opts[@]}" \
      $data_dir/train_si284 $data_dir/lang_nosp $exp_dir/mono0a_ali \
      $exp_dir/hybrid_$mfcc || exit 1
  fi
  
  if $decode; then

    for x in test_eval92; do 
      local_pyspeech/fetch_feats.sh $data_dir/$x $mfcc || exit 1;
    done

    [ ! -d $exp_dir/hybrid_$mfcc/decode_nosp_tgpr_eval92 ] \
      && mkdir -p $exp_dir/hybrid_$mfcc/decode_nosp_tgpr_eval92

    decode_extra_opts=(--num-threads 6)
    
    steps/nnet2/decode.sh --cmd "$decode_cmd" \
      --nj 8 \
      "${decode_extra_opts[@]}" \
      $exp_dir/mono0a/graph_nosp_tgpr \
      $data_dir/test_eval92 \
      $exp_dir/hybrid_$mfcc/decode_nosp_tgpr_eval92 | \
      tee $exp_dir/hybrid_$mfcc/decode_nosp_tgpr_eval92/decode.log
  fi
fi
