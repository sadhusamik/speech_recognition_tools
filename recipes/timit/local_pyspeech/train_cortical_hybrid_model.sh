#!/usr/bin/env bash

. ./path.sh
. ./src_paths.sh

stage=0
nj=10
hybrid_dir=exp/hybrid_corticalmodel
data_dir=data
lang_dir=lang_test_bg
feat_type=mfcc
hmm_dir=exp/tri3
use_gpu=true
train_set=train
dev_set=dev
aux_dev_set=None
test_set=test
nn_name=nnet_cortical
num_egs_jobs=10

# Neural network config
num_layers_A1=2
num_layers_A3=3
hidden_dim_A1=512
hidden_dim_A3=50
batch_size=64
epochs=300
num_classes=38
model_save_interval=10
dropout=0
weight_decay=0
learning_rate=0.0001
in_channels="1,128,128"
out_channels="128,128,128"
kernel="3,5"
hidden_size=1024
l_num_layers=0
d_num_layers=2
load_model=None
model_type=phonotopic

# Feature config
feature_dim=13
left_context=4
right_context=4
max_seq_len=512
ali_type="phone"
ali_append=
per_utt_cmvn=false
data_prep_only=false
nnet_arch=rnn

. utils/parse_options.sh || exit 1;

mkdir -p $hybrid_dir
log_dir=$hybrid_dir/log


run_script="train_A1A3model.py --load_model=$load_model --num_layers_A1=$num_layers_A1 --num_layers_A3=$num_layers_A3 --hidden_dim_A1=$hidden_dim_A1 --hidden_dim_A3=$hidden_dim_A3"

echo "$0: nn_name=$nn_name"

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

  for x in $train_set $dev_set $test_set; do
    egs_dir=$hybrid_dir/egs/$x
    cmvn_path=$hybrid_dir/perutt_cmvn_${x}_${feat_type}.scp
    mkdir -p $egs_dir
    if [ ! -z $ali_append ]; then
      ali_name=${ali_append}_${x}
    else
      ali_name=$x
    fi

    python3 $nnet_src/data_prep_for_seq.py \
      --num_jobs=$num_egs_jobs \
      --feat_type=$cmvn_type,$cmvn_path \
      --ali_type=$ali_type \
      --max_seq_len=$max_seq_len \
      --concat_feats=${left_context},${right_context} \
      $data_dir/$x/feats.scp \
      ${hmm_dir}_ali_${ali_name} \
      $egs_dir || exit 1;
  done
fi

if $data_prep_only; then
  exit
fi

if [ $stage -le 1 ]; then
  if $use_gpu; then
    $cuda_cmd --mem 5G \
      $hybrid_dir/log/train_rnn_${nn_name}.log \
      python3 $nnet_src/$run_script \
      --use_gpu \
      --model_type=$model_type \
      --train_set=$train_set \
      --dev_set=$dev_set \
      --aux_dev_set=$aux_dev_set \
      --batch_size=$batch_size \
      --epochs=$epochs \
      --dropout=$dropout \
      --learning_rate=$learning_rate \
      --weight_decay=$weight_decay \
      --feature_dim=$feature_dim \
      --num_classes=$num_classes \
      --model_save_interval=$model_save_interval \
      --experiment_name=exp_1 \
      $hybrid_dir/egs \
      $hybrid_dir/$nn_name || exit 1;
  else

    queue.pl --mem 5G \
      $hybrid_dir/log/train_rnn_${nn_name}.log \
      python3 $nnet_src/$run_script \
      --model_type=$model_type \
      --train_set=$train_set \
      --dev_set=$dev_set \
      --aux_dev_set=$aux_dev_set \
      --batch_size=$batch_size \
      --epochs=$epochs \
      --dropout=$dropout \
      --learning_rate=$learning_rate \
      --weight_decay=$weight_decay \
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

