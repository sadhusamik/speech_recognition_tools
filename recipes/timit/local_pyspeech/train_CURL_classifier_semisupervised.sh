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
train_set_ali=train
dev_set=dev
nn_name=nnet_gru_3lenc_1lclas_1lae_256nodes
num_egs_jobs=10

# Neural network config
encoder_num_layers=2
decoder_num_layers=2
classifier_num_layers=1
hidden_dim=256
hidden_dim_classifier=256

bn_dim=30
batch_size=64
epochs=200
comp_num=25
num_classes=116
model_save_interval=10
weight_decay=0.001
per_utt_cmvn=false

# Feature config
feature_dim=13
left_context=4
right_context=4
max_seq_len=512
ali_type="phone"


. utils/parse_options.sh || exit 1;

mkdir -p $hybrid_dir 
log_dir=$hybrid_dir/log

echo "$0: nn_name=$nn_name"

if [ $stage -le 0 ]; then 

  if $per_utt_cmvn; then
    cmvn_type=cmvn_utt
    for x in $train_set $test_set $dev_set; do
      cmvn_path=$hybrid_dir/perutt_cmvn_${x}_${feat_type}
      compute-cmvn-stats \
        scp:$data_dir/$x/feats.scp \
        ark,scp:$cmvn_path.ark,$cmvn_path.scp  || exit 1;
    done
  else
    cmvn_type=cmvn
    cmvn_path=$hybrid_dir/global_cmvn_${feat_type}
    compute-cmvn-stats scp:$data_dir/$train_set/feats.scp $cmvn_path  || exit 1;
  fi


  # Dataprep training set
  egs_dir=$hybrid_dir/egs/${train_set}
  mkdir -p $egs_dir
  cmvn_path=$hybrid_dir/perutt_cmvn_${train_set}_${feat_type}.scp
  python3 $nnet_src/data_prep_semisup.py \
    --num_jobs=$num_egs_jobs \
    --feat_type=$cmvn_type,$cmvn_path \
    --ali_type=$ali_type \
    --max_seq_len=$max_seq_len \
    --concat_feats=${left_context},${right_context} \
    $data_dir/${train_set}/feats.scp \
    ${hmm_dir}_ali_${train_set_ali} \
    $egs_dir || exit 1;
  
  # Data prep dev set  
  egs_dir=$hybrid_dir/egs/${dev_set}
  mkdir -p $egs_dir
  cmvn_path=$hybrid_dir/perutt_cmvn_${dev_set}_${feat_type}.scp
  python3 $nnet_src/data_prep_semisup.py \
    --num_jobs=$num_egs_jobs \
    --feat_type=$cmvn_type,$cmvn_path \
    --ali_type=$ali_type \
    --max_seq_len=$max_seq_len \
    --concat_feats=${left_context},${right_context} \
    $data_dir/${dev_set}/feats.scp \
    ${hmm_dir}_ali_${dev_set} \
    $egs_dir || exit 1;
  
fi

if [ $stage -le 1 ]; then 
  if $use_gpu; then 
    $cuda_cmd --mem 5G \
      $hybrid_dir/log/train_CURL_classifier_${nn_name}.log \
      python3 $nnet_src/train_CURL_classifier.py \
      --use_gpu \
      --train_set=$train_set \
      --dev_set=$dev_set \
      --encoder_num_layers=$encoder_num_layers \
      --decoder_num_layers=$decoder_num_layers \
      --classifier_num_layers=$classifier_num_layers \
      --hidden_dim=$hidden_dim \
      --hidden_dim_classifier=$hidden_dim_classifier \
      --bn_dim=$bn_dim \
      --comp_num=$comp_num \
      --num_classes=$num_classes \
      --batch_size=$batch_size \
      --epochs=$epochs \
      --weight_decay=$weight_decay \
      --feature_dim=$feature_dim \
      --model_save_interval=$model_save_interval \
      --experiment_name=exp_1 \
      $hybrid_dir/egs \
      $hybrid_dir/$nn_name || exit 1;
  else

    queue.pl --mem 5G \
      $hybrid_dir/log/train_CURL_classifier_${nn_name}.log \
      python3 $nnet_src/train_CURL_classifier.py \
      --train_set=$train_set \
      --dev_set=$dev_set \
      --encoder_num_layers=$encoder_num_layers \
      --decoder_num_layers=$decoder_num_layers \
      --classifier_num_layers=$classifier_num_layers \
      --hidden_dim=$hidden_dim \
      --hidden_dim_classifier=$hidden_dim_classifier \
      --bn_dim=$bn_dim \
      --comp_num=$comp_num \
      --num_classes=$num_classes \
      --batch_size=$batch_size \
      --epochs=$epochs \
      --weight_decay=$weight_decay \
      --feature_dim=$feature_dim \
      --model_save_interval=$model_save_interval \
      --experiment_name=exp_1 \
      $hybrid_dir/egs \
      $hybrid_dir/$nn_name || exit 1;
  fi

fi

