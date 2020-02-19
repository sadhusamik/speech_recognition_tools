#!/usr/bin/env bash

. ./path.sh
. ./src_paths.sh 

stage=0
nj=10
hybrid_dir=exp/hybrid_generative_pytorch
data_dir=data
lang_dir=lang_test_bg
feat_type=mfcc
hmm_dir=exp/tri3
use_gpu=true
train_set=train
dev_set=dev
test_set=test
nn_name=nnet_gru_3lenc_1lclas_1lae_256nodes
num_egs_jobs=2

# Neural network config
encoder_num_layers=3
classifier_num_layers=1
ae_num_layers=1
hidden_dim=256
bn_dim=30
batch_size=64
epochs=300
num_classes=38
model_save_interval=10

# Feature config
feature_dim=13
left_context=4
right_context=4
max_seq_len=512
ali_type="phone"


. utils/parse_options.sh || exit 1;

mkdir -p $hybrid_dir 
log_dir=$hybrid_dir/log

if [ $stage -le 0 ]; then 
  for x in $train_set $dev_set ; do 
    local_pyspeech/fetch_feats.sh $data_dir/$x $feat_type || exit 1;
  done

  cmvn_path=$hybrid_dir/global_cmvn
  compute-cmvn-stats scp:$data_dir/$train_set/feats.scp $cmvn_path  || exit 1;

  for x in $train_set $dev_set ; do
    egs_dir=$hybrid_dir/egs/$x
    mkdir -p $egs_dir
    python3 $nnet_src/data_prep_for_seq.py \
      --num_jobs=$num_egs_jobs \
      --feat_type=cmvn,$cmvn_path \
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
      $hybrid_dir/log/train_feedgen.log \
      python3 $nnet_src/train_feedforward_generative_model.py \
      --use_gpu \
      --train_set=$train_set \
      --dev_set=$dev_set \
      --encoder_num_layers=$encoder_num_layers \
      --classifier_num_layer=$classifier_num_layers \
      --ae_num_layers=$ae_num_layers \
      --hidden_dim=$hidden_dim \
      --bn_dim=$bn_dim \
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
      $hybrid_dir/log/train_feedgen.log \
      python3 $nnet_src/train_feedforward_generative_model.py \
      --train_set=$train_set \
      --dev_set=$dev_set \
      --encoder_num_layers=$encoder_num_layers \
      --classifier_num_layer=$classifier_num_layers \
      --ae_num_layers=$ae_num_layers \
      --hidden_dim=$hidden_dim \
      --bn_dim=$bn_dim \
      --batch_size=$batch_size \
      --epochs=$epochs \
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

