#!/bin/bash

## Train a Hybrid Pytorch Model with best Kaldi alignments
# Samik Sadhu 

stage=2

exp_dir=exp_hybrid
data_dir=data
mfcc=mfcc
hmm_dir=exp/tri5b
wsj_dir=../wsj

. ./cmd.sh
. ./path.sh
. ./src_paths.sh

. utils/parse_options.sh


if [ $stage -le 1 ]; then
  
  echo "## STAGE: MFCC feature extraction ##"

  for x in dev_clean test_clean dev_other test_other train_clean_100; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 data/$x || exit 1;
    steps/compute_cmvn_stats.sh data/$x || exit 1;  
  done
  
fi

for x in dev_clean test_clean dev_other test_other train_clean_360; do
  local_pyspeech/fetch_feats.sh --feat_source kaldi \
    data/$x mfcc mfcc || exit 1;
done

if [ $stage -le 2 ]; then

  train=true
  decode=true

  echo "## Train RNN Hybrid Model ##"
  
  if $train; then
    local_pyspeech/train_rnn_hybrid.sh \
      --stage 2 \
      --use_gpu  true \
      --data_dir $data_dir \
      --hybrid_dir $exp_dir/hybrid_triphone_rnn \
      --feat_type $mfcc \
      --hmm_dir $hmm_dir \
      --train_set train_clean_360 \
      --dev_set dev_clean \
      --test_set test_clean \
      --nn_name nnet_triphone_4l_300nodes_bigger \
      --num_layers 4 \
      --left_context 4 \
      --right_context 4 \
      --ali_type "pdf" \
      --ali_append "wsj" \
      --per_utt_cmvn true \
      --hidden_dim 300 \
      --batch_size 64 \
      --epochs 100 \
      --num_classes 3376 \
      --feature_dim 13 || exit 1 ;
  fi 

  if $decode; then 

    for testset in test_clean test_other; do
        local_pyspeech/decode_dnn.sh \
          --stage 0 \
          --nj 50 \
          --pw 0.5 \
          --ae_type "noae" \
          --model_iter 50 \
          --append "nnet_triphone_4l_300nodes_bigger_iter50" \
          --score_script score_wsj.sh \
          --hybrid_dir $exp_dir/hybrid_triphone_rnn \
          --lang_dir $wsj_dir/data/lang_nosp_test_tgpr \
          --feat_type $mfcc \
          --hmm_dir $wsj_dir/exp/tri3b \
          --graph_name graph \
          --test_set $testset \
          --train_set train_clean_360 \
          --nn_name nnet_triphone_4l_300nodes_bigger || exit 1;
     done
  fi
fi


if [ $stage -le 3 ]; then 

  echo " Train p(x| data) model for Librispeech "
    local_pyspeech/train_VAE.sh \
      --stage 1 \
      --use_gpu  true \
      --per_utt_cmvn true \
      --data_dir data \
      --hybrid_dir $exp_dir/hybrid_triphone_rnn \
      --feat_type mfcc \
      --hmm_dir $hmm_dir \
      --train_set train_clean_360 \
      --dev_set test_clean \
      --nn_name LIBRISPEECH_px_VAE_enc1l_dec1l_bigger_300nodes \
      --encoder_num_layers 1 \
      --decoder_num_layers 1 \
      --left_context 4 \
      --right_context 4 \
      --weight_decay 0 \
      --ali_type "pdf" \
      --hidden_dim 300 \
      --bn_dim 100 \
      --batch_size 64 \
      --epochs 300 \
      --feature_dim 13 || exit 1 ;
fi

exit
if [ $stage -le 4 ]; then 

  echo " Train posterior VAE-PM for LIBRI "
    local_pyspeech/train_posterior_VAE.sh \
      --stage 1 \
      --use_gpu  true \
      --per_utt_cmvn true \
      --data_dir data \
      --hybrid_dir $exp_dir/hybrid_triphone_rnn \
      --feat_type mfcc \
      --hmm_dir $hmm_dir \
      --train_set train_clean_360 \
      --dev_set test_clean \
      --nn_name LIBRI_post_VAE_enc1l_dec1l_300nodes \
      --nnet_model "exp_hybrid/hybrid_triphone_rnn/nnet_triphone_4l_300nodes_bigger/exp_1.dir/exp_1__epoch_50.model"\
      --encoder_num_layers 1 \
      --decoder_num_layers 1 \
      --left_context 4 \
      --right_context 4 \
      --weight_decay 0 \
      --ali_type "pdf" \
      --hidden_dim 512 \
      --bn_dim 150 \
      --batch_size 64 \
      --epochs 300 \
      --feature_dim 13 || exit 1 ;
fi
