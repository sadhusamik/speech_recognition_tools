#!/bin/bash

## Train a Hybrid Pytorch Model with best Kaldi alignments
# Samik Sadhu 

stage=0

exp_dir=exp_hybrid
data_dir=data
mfcc=mfcc
hmm_dir=exp/tri3
wsj_dir=../wsj

. ./cmd.sh
. ./path.sh
. ./src_paths.sh

. utils/parse_options.sh


if [ $stage -le 0 ]; then 
  train=false
  decode=false
  decode_real=true
  decode_unilang=false

  echo "## Train RNN Hybrid Model ##"
  
  if $train; then
    local_pyspeech/train_rnn_hybrid.sh \
      --stage 2 \
      --use_gpu  true \
      --data_dir $data_dir \
      --hybrid_dir $exp_dir/hybrid_triphone_rnn \
      --feat_type $mfcc \
      --hmm_dir $hmm_dir \
      --train_set tr_simu_8ch \
      --dev_set dt_simu_8ch \
      --test_set et_simu_8ch \
      --nn_name nnet_triphone_4l_300nodes \
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

    for testset in et_simu_1ch et_real_1ch; do
        local_pyspeech/decode_dnn.sh \
          --stage 0 \
          --nj 20 \
          --pw 0.5 \
          --ae_type "noae" \
          --model_iter 20 \
          --append "nnet_triphone_4l_300nodes_iter20" \
          --score_script score_wsj.sh \
          --hybrid_dir $exp_dir/hybrid_triphone_rnn \
          --lang_dir $wsj_dir/data/lang_nosp_test_tgpr \
          --override_prior exp_hybrid/hybrid_triphone_rnn/priors \
          --feat_type $mfcc \
          --hmm_dir $wsj_dir/exp/tri3b \
          --graph_name graph \
          --test_set $testset \
          --train_set tr_simu_8ch \
          --nn_name nnet_triphone_4l_300nodes || exit 1;
      done
  fi
fi

if [ $stage -le 1 ]; then 

  echo " Train p(x| data) model for Reverb "
    local_pyspeech/train_VAE.sh \
      --stage 1 \
      --use_gpu  true \
      --per_utt_cmvn true \
      --data_dir data \
      --hybrid_dir $exp_dir/hybrid_triphone_rnn \
      --feat_type mfcc \
      --hmm_dir $hmm_dir \
      --train_set tr_simu_8ch \
      --dev_set dt_simu_8ch \
      --nn_name REVERB_px_VAE_enc1l_dec1l_300nodes \
      --encoder_num_layers 1 \
      --decoder_num_layers 1 \
      --left_context 4 \
      --right_context 4 \
      --weight_decay 0 \
      --ali_type "pdf" \
      --ali_append "wsj" \
      --hidden_dim 300 \
      --bn_dim 100 \
      --batch_size 64 \
      --epochs 300 \
      --feature_dim 13 || exit 1 ;
fi

exit

if [ $stage -le 4 ]; then 

  echo " Train posterior VAE-PM for REVERB "
    local_pyspeech/train_posterior_VAE.sh \
      --stage 1 \
      --use_gpu  true \
      --per_utt_cmvn true \
      --data_dir data \
      --hybrid_dir $exp_dir/hybrid_triphone_rnn \
      --feat_type mfcc \
      --hmm_dir $hmm_dir \
      --train_set tr_simu_8ch \
      --dev_set dt_simu_8ch \
      --nn_name REVERB_post_VAE_enc1l_dec1l_300nodes \
      --nnet_model "exp_hybrid/hybrid_triphone_rnn/nnet_triphone_4l_300nodes/exp_1.dir/exp_1__epoch_20.model"\
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
