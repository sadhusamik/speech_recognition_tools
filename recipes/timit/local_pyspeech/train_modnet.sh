#!/usr/bin/env bash

. ./path.sh
. ./src_paths.sh 

stage=0
nj=10
hybrid_dir=exp/hybrid_rnn_pytorch
data_dir=data
lang_dir=lang_test_bg
feat_type=mfcc
hmm_dir=exp/tri3
use_gpu=true
train_set=train
dev_set=dev
nn_name=nnet_gru_4ly_256nodes
num_egs_jobs=10

# Neural network config
num_layers_dec=3
hidden_dim=256
batch_size=64
epochs=300
num_classes=38
in_channels="1,20,20"
out_channels="20,20,20"
kernel=3
input_filter_kernel=10
freq_num=15
head_num=30
wind_size=0.5
model_save_interval=10

# Feature config
feature_dim=13
left_context=4
right_context=4
max_seq_len=512
ali_type="phone"
init_mod=true
per_utt_cmvn=true

. utils/parse_options.sh || exit 1;

echo "$0: nn_name=$nn_name"
mkdir -p $hybrid_dir 
log_dir=$hybrid_dir/log

if [ $stage -le 0 ]; then 

  if $per_utt_cmvn; then
    cmvn_type=cmvn_utt
    for x in $train_set $test_set $dev_set; do
      cmvn_path=`realpath $hybrid_dir/perutt_cmvn_${x}_${feat_type}`
      compute-cmvn-stats \
        scp:$data_dir/$x/feats.scp \
        ark,scp:$cmvn_path.ark,$cmvn_path.scp  || exit 1;
    done
  else
    cmvn_type=cmvn
    cmvn_path=`realpath $hybrid_dir/global_cmvn_${feat_type}`
    compute-cmvn-stats scp:$data_dir/$train_set/feats.scp $cmvn_path  || exit 1;
  fi

  for x in $train_set $dev_set ; do
    egs_dir=$hybrid_dir/egs/$x
    mkdir -p $egs_dir
    cmvn_path=$hybrid_dir/perutt_cmvn_${x}_${feat_type}.scp
    python3 $nnet_src/data_prep_for_seq.py \
      --num_jobs=$num_egs_jobs \
      --feat_type=$cmvn_type,$cmvn_path \
      --ali_type=$ali_type \
      --max_seq_len=$max_seq_len \
      --concat_feats=${left_context},${right_context} \
      $data_dir/$x/feats.scp \
      ${hmm_dir}_ali_${x} \
      $egs_dir || exit 1;
  done
fi

if [ $stage -le 1 ]; then 
  if $use_gpu; then 
    $cuda_cmd --mem 5G \
      $hybrid_dir/log/train_modnet_${nn_name}.log \
      python3 $nnet_src/train_modnet.py \
      --use_gpu \
      --train_set=$train_set \
      --dev_set=$dev_set \
      --num_layers_dec=$num_layers_dec \
      --hidden_dim=$hidden_dim \
      --in_channels=$in_channels \
      --out_channel=$out_channels \
      --kernel=$kernel \
      --input_filter_kernel=$input_filter_kernel \
      --freq_num=$freq_num \
      --wind_size=$wind_size \
      --head_num=$head_num \
      --batch_size=$batch_size \
      --epochs=$epochs \
      --feature_dim=$feature_dim \
      --num_classes=$num_classes \
      --model_save_interval=$model_save_interval \
      --experiment_name=exp_1 \
      $hybrid_dir/egs \
      $hybrid_dir/$nn_name || exit 1;
  else

    queue.pl --mem 5G \
      $hybrid_dir/log/train_modnet_${nn_name}.log \
      python3 $nnet_src/train_modnet.py \
      --train_set=$train_set \
      --dev_set=$dev_set \
      --num_layers_dec=$num_layers_dec \
      --hidden_dim=$hidden_dim \
      --in_channels=$in_channels \
      --out_channel=$out_channels \
      --kernel=$kernel \
      --input_filter_kernel=$input_filter_kernel \
      --freq_num=$freq_num \
      --wind_size=$wind_size \
      --head_num=$head_num \
      --batch_size=$batch_size \
      --epochs=$epochs \
      --feature_dim=$feature_dim \
      --num_classes=$num_classes \
      --model_save_interval=$model_save_interval \
      --experiment_name=exp_1 \
      $hybrid_dir/egs \
      $hybrid_dir/$nn_name || exit 1;
  fi

fi

if [ $stage -le 2 ]; then 
  queue.pl $hybrid_dir/log/compute_prior.log \
    python3 $nnet_src/compute_log_prior.py \
    --ali_type=$ali_type \
    --num_classes=$num_classes \
    ${hmm_dir}_ali_${train_set} \
    $hybrid_dir/priors || exit 1;
fi

