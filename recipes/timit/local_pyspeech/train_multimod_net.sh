#!/usr/bin/env bash

. ./path.sh
. ./src_paths.sh 

stage=0
nj=10
hybrid_dir=exp/hybrid_generative_pytorch
data_dir=data
lang_dir=lang_test_bg
feat_types=mfcc
hmm_dir=exp/tri3
use_gpu=true
train_set=train
dev_set=dev
test_set=test
nn_name=nnet_gru_3lenc_1lclas_1lae_256nodes
num_egs_jobs=2

# Neural network config
num_layers_subband=1
num_layers=1
mod_num=3
hidden_dim_subband=100
batch_size=64
epochs=300
num_classes=38
model_save_interval=10
weight_decay=0

# Feature config
feature_dim=13
left_context=0
right_context=0
max_seq_len=512
ali_type="phone"
data_prep_only=false

. utils/parse_options.sh || exit 1;

echo "$0:nn_name: $nn_name"
mkdir -p $hybrid_dir 
log_dir=$hybrid_dir/log

if [ $stage -le 0 ]; then
  for f in $feat_types; do 
    for x in $train_set $dev_set ; do 
      local_pyspeech/generate_feats_scp.sh $data_dir/$x $f || exit 1;
    done
  done
  
  for f in $feat_types; do
    cmvn_path=$hybrid_dir/global_cmvn_$f
    compute-cmvn-stats scp:$data_dir/$train_set/$f/feats.scp $cmvn_path  || exit 1;
  done

  for x in $train_set $dev_set ; do
    for f in $feat_types; do
      cmvn_path=$hybrid_dir/global_cmvn_$f
      egs_dir=$hybrid_dir/egs/${x}/${f}
      mkdir -p $egs_dir
      python3 $nnet_src/data_prep_for_seq.py \
        --num_jobs=$num_egs_jobs \
        --feat_type=cmvn,$cmvn_path \
        --ali_type=$ali_type \
        --max_seq_len=$max_seq_len \
        --concat_feats=${left_context},${right_context} \
        $data_dir/$x/$f/feats.scp \
        ${hmm_dir}_ali_${x} \
        $egs_dir || exit 1;
    done
  done
fi

#if $data_prep_only; then 
exit
#fi

if [ $stage -le 1 ]; then 
  if $use_gpu; then 
    $cuda_cmd --mem 5G \
      $hybrid_dir/log/train_multimod_${nn_name}.log \
      python3 $nnet_src/train_multimod_nnet.py \
      --use_gpu \
      --train_set=$train_set \
      --dev_set=$dev_set \
      --subband_sets="$feat_types" \
      --num_layers_subband=$num_layers_subband \
      --hidden_dim_subband=$hidden_dim_subband \
      --num_layers=$num_layers \
      --mod_num=$mod_num \
      --batch_size=$batch_size \
      --epochs=$epochs \
      --weight_decay=$weight_decay \
      --feature_dim=$feature_dim \
      --num_classes=$num_classes \
      --model_save_interval=$model_save_interval \
      --experiment_name=exp_1 \
      $hybrid_dir/egs \
      $hybrid_dir/$nn_name || exit 1;
  else

    queue.pl --mem 5G \
      $hybrid_dir/log/train_multimod_${nn_name}.log \
      python3 $nnet_src/train_multimod_nnet.py \
      --train_set=$train_set \
      --dev_set=$dev_set \
      --subband_sets="$feat_types" \
      --num_layers_subband=$num_layers_subband \
      --hidden_dim_subband=$hidden_dim_subband \
      --num_layers=$num_layers \
      --mod_num=$mod_num \
      --batch_size=$batch_size \
      --epochs=$epochs \
      --weight_decay=$weight_decay \
      --feature_dim=$feature_dim \
      --num_classes=$num_classes \
      --model_save_interval=$model_save_interval \
      --experiment_name=exp_1 \
      $hybrid_dir/egs \
      $hybrid_dir/$nn_name || exit 1;
  fi

  cp $hybrid_dir/$nn_name/exp_1.dir/exp_1__epoch_300.model \
    $hybrid_dir/$nn_name/exp_1.dir/final.mdl || exit 1 ;
fi

if [ $stage -le 2 ]; then 
  queue.pl $hybrid_dir/log/compute_prior.log \
    python3 $nnet_src/compute_log_prior.py \
    --ali_type=$ali_type \
    --num_classes=$num_classes \
    ${hmm_dir}_ali_${train_set} \
    $hybrid_dir/priors || exit 1;
fi

