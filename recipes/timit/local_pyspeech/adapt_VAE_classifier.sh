#!/bin/bash 

. ./path.sh
. ./src_paths.sh

stage=0
num_jobs=5
left_context=4
right_context=4
ali_type=pdf
max_seq_len=512
nnet_name=nnet_gru_3lenc_1lclas_1lae_256nodes
model_save_interval=10 

exp_dir=exp
hybrid_dir=hybrid_monophone_generative_pytorch
adapt_name=adapt
adapt_cmvn_path=
model_iter=
use_gpu=true
adapt_model_name=adapt

# Datasets
anchor_set=train
anchor_scp=data/train/mfcc/feats.scp
anchor_ali=exp/mono_ali_train

adapt_set=train_si84
adapt_scp=../wsj/data/train_si284/feats.scp
adapt_ali=../wsj/exp/mono0a_ali_train_si284

test_set=test_eval92
test_scp=../wsj/data/test_eval92/feats.scp
test_ali=../wsj/exp/mono0a_ali_test_eval92

#Adaptation Parameters
batch_size=64
mm_weight=1
adapt_weight=200
anchor_weight=1
epochs=500 

. utils/parse_options.sh

adapt_dir=$exp_dir/$adapt_name
mkdir -p adapt_dir
if [ -z $model_iter ]; then 
  echo "$0: Choosing best model"
  model_init="$exp_dir/$hybrid_dir/$nnet_name/exp_1.dir/final.mdl"
else
  model_init="$exp_dir/$hybrid_dir/$nnet_name/exp_1.dir/exp_1__epoch_${model_iter}.model"
fi
cmvn_path=$exp_dir/$hybrid_dir/global_cmvn 

if [ -z $adapt_cmvn_path ]; then 
  adapt_cmvn_path=$cmvn_path
fi

if [ $stage -le 0 ]; then
  
  # Anchor examples
  egs_dir=$adapt_dir/egs/$anchor_set
  mkdir -p $egs_dir
  (
  python3 $nnet_src/data_prep_for_seq.py \
    --num_jobs=5 \
    --feat_type=cmvn,$cmvn_path \
    --concat_feats=$left_context,$right_context \
    --ali_type=$ali_type \
    --max_seq_len=$max_seq_len \
    $anchor_scp \
    $anchor_ali \
    $egs_dir || exit 1;
  ) &

  # Adapt Examples
  egs_dir=$adapt_dir/egs/$adapt_set
  mkdir -p $egs_dir
  (
  python3 $nnet_src/data_prep_for_seq.py \
    --num_jobs=5 \
    --feat_type=cmvn,$adapt_cmvn_path \
    --concat_feats=$left_context,$right_context \
    --ali_type=$ali_type \
    --max_seq_len=$max_seq_len \
    $adapt_scp \
    $adapt_ali \
    $egs_dir || exit 1;
  ) &

  egs_dir=$adapt_dir/egs/$test_set
  mkdir -p $egs_dir
  (
  python3 $nnet_src/data_prep_for_seq.py \
    --num_jobs=5 \
    --feat_type=cmvn,$adapt_cmvn_path \
    --concat_feats=$left_context,$right_context \
    --ali_type=$ali_type \
    --max_seq_len=$max_seq_len \
    $test_scp \
    $test_ali \
    $egs_dir || exit 1;
  ) &
  wait;
fi

if [ $stage -le 1 ]; then 
  egs_dir=$adapt_dir/egs
  if $use_gpu ; then
    $cuda_cmd --mem 5G $adapt_dir/$adapt_model_name.log \
      python3 $nnet_src/nnet_adapt_VAE_classifier.py \
      --use_gpu \
      --anchor_set=$anchor_set \
      --adapt_set=$adapt_set \
      --test_set=$test_set \
      --batch_size=$batch_size \
      --mm_weight=$mm_weight \
      --adapt_weight=$adapt_weight \
      --epochs=$epochs \
      --anchor_weight=$anchor_weight \
      --model_save_interval=$model_save_interval \
      $model_init \
      $egs_dir \
      $adapt_dir/$adapt_model_name || exit 1;
  else
    queue.pl --mem 5G $adapt_dir/$adapt_model_name.log \
      python3 $nnet_src/nnet_adapt_VAE_classifier.py \
      --anchor_set=$anchor_set \
      --adapt_set=$adapt_set \
      --test_set=$test_set \
      --batch_size=$batch_size \
      --mm_weight=$mm_weight \
      --adapt_weight=$adapt_weight \
      --epochs=$epochs \
      --anchor_weight=$anchor_weight \
      --model_save_interval=$model_save_interval \
      $model_init \
      $egs_dir \
      $adapt_dir/$adapt_model_name || exit 1;
  fi
fi
exit
if $decode; then 
local_pyspeech/decode_dnn.sh \
  --stage 0 \
  --nj 10 \
  --pw 0.5 \
  --append "adapted" \
  --model "exp_hybrid/adapt_mono_genclass_train_si84/adapt.dir/exp_run.dir/exp_run__epoch_50.model" \
  --score_script score_wsj.sh \
  --hybrid_dir $exp_dir/hybrid_monophone_generative_pytorch \
  --data_dir ../wsj/data/ \
  --lang_dir ../wsj/data/lang_nosp_test_tg \
  --feat_type mfcc \
  --hmm_dir exp/mono \
  --graph_name graph_new \
  --test_set test_eval92 \
  --nn_name nnet_gru_3lenc_1lclas_1lae_256nodes || exit 1;
fi
