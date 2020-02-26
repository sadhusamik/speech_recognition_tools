#!/bin/bash

## Train a Hybrid Pytorch Model with best Kaldi alignments
# Samik Sadhu 

stage=0

exp_dir=exp_hybrid
data_dir=data
mfcc=mfcc

. ./cmd.sh
. ./path.sh
. ./src_paths.sh

. utils/parse_options.sh

if [ $stage -le 0 ]; then 

  echo "## Get alignment for hybrid model ##"
  ./run_get_hq_ali.sh || exit 1;
fi

# mvector parameters

nfilters=20
coeff_0=1
coeff_n=15
fduration=0.5
order=30

modspec=modspec_${nfilters}_${coeff_0}_${coeff_n}_${fduration}_${order}

if [ $stage -le 1 ]; then 
  
  echo "Modulation feature extraction"

  for x in train test dev ; do
    local_pyspeech/make_modspec_feats.sh --nj $nj_modspec \
      --nfilters $nfilters \
      --coeff_0 $coeff_0 \
      --coeff_n $coeff_n \
      --fduration $fduration \
      --order $order \
      $data_dir/$x \
      $data_dir/$x/$modspec || exit 1;

    local_pyspeech/get_cmvn.sh \
      $data_dir/$x $data_dir/$x/$modspec/cmvn || exit 1;

    local_pyspeech/generate_feats_scp.sh $data_dir/$x $modspec || exit 1

  done

fi

if [ $stage -le 2 ]; then 
  train=true
  decode=true

  echo "## Train RNN Hybrid Model ##"
  
  if $train; then
    local_pyspeech/train_rnn_hybrid.sh \
      --stage 1 \
      --use_gpu  true \
      --data_dir $data_dir \
      --hybrid_dir $exp_dir/hybrid_monophone \
      --feat_type $modspec \
      --hmm_dir exp/mono \
      --train_set train \
      --dev_set dev \
      --test_set test \
      --nn_name nnet_mono_3l_300nodes_drop0_wd0_initlr0.001_lrtol0.5_reverted \
      --num_layers 3 \
      --left_context 0\
      --right_context 0 \
      --ali_type "pdf" \
      --hidden_dim 300 \
      --batch_size 64 \
      --epochs 100 \
      --num_classes 114 \
      --feature_dim 225 || exit 1 ;
  fi 

  if $decode_wsj; then 

    local_pyspeech/decode_dnn.sh \
      --stage 0 \
      --nj 10 \
      --pw 0.5 \
      --ae_type "noae" \
      --model_iter 80 \
      --append "nnet_mono_3l_300nodes_drop0_wd0_initlr0.001_lrtol0.5_reverted_iter80" \
      --score_script score.sh \
      --hybrid_dir $exp_dir/hybrid_monophone_generative_pytorch \
      --lang_dir data/lang_test_bg \
      --feat_type $modspec \
      --hmm_dir exp/mono \
      --graph_name graph \
      --test_set test \
      --nn_name nnet_mono_3l_300nodes_drop0_wd0_initlr0.001_lrtol0.5_reverted || exit 1;
  fi
fi
