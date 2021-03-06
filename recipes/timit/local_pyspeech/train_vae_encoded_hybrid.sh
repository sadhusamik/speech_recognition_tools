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
nn_name=nnet_gru_3lenc_1lclas_1lae_256nodes
num_egs_jobs=2

# Neural network config
num_layers=1
hidden_size=300
batch_size=64
epochs=100
num_classes=38
model_save_interval=10
dropout=0
weight_decay=0
vae=vae
per_utt_cmvn==true
in_channels="128,128,128"
out_channels="128,128,128"
kernel="3,9"

# Feature config
feature_dim=13
left_context=4
right_context=4
max_seq_len=512
ali_type="phone"
ali_append=
vae_type="modulation"
hybrid_arch=cnn
l_num_layers=2
d_num_layers=2
num_streams=15

. utils/parse_options.sh || exit 1;

mkdir -p $hybrid_dir 
log_dir=$hybrid_dir/log

echo "$0: nn_name=$nn_name"

case $hybrid_arch in 
  cldnn3d)
    run_script="train_CNNVAE_encoded_nnet_classifier_cldnn3d.py --num_streams=$num_streams --in_channels=$in_channels --out_channels=$out_channels --kernel=$kernel --hidden_size=$hidden_size --l_num_layers=$l_num_layers --d_num_layers=$d_num_layers"
    ;;
  cldnn)
    run_script="train_CNNVAE_encoded_nnet_classifier_cldnn.py --in_channels=$in_channels --out_channels=$out_channels --kernel=$kernel --hidden_size=$hidden_size --l_num_layers=$l_num_layers --d_num_layers=$d_num_layers"
    ;;
  cnn)
    run_script="train_CNNVAE_encoded_nnet_classifier_cnn.py --in_channels=$in_channels --out_channels=$out_channels --kernel=$kernel"
    ;;
  rnn)
    run_script=train_VAE_encoded_nnet_classfier.py
    ;;
  *) 
    echo "$0: Hybrid Arch $hybrid_arch is not valid!"
    exit 1;
esac

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
  
  for x in $train_set $dev_set ; do
    egs_dir=$hybrid_dir/egs/$x
    mkdir -p $egs_dir
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

if [ $stage -le 1 ]; then 
  if $use_gpu; then 
    $cuda_ccmd --mem 2G \
      $hybrid_dir/log/train_vae_encoded_nnet_${nn_name}.log \
      python3 $nnet_src/$run_script \
      --use_gpu \
      --train_set=$train_set \
      --vae_type=$vae_type \
      --dev_set=$dev_set \
      --batch_size=$batch_size \
      --epochs=$epochs \
      --weight_decay=$weight_decay \
      --num_classes=$num_classes \
      --model_save_interval=$model_save_interval \
      --experiment_name=exp_1 \
      $vae \
      $hybrid_dir/egs \
      $hybrid_dir/$nn_name || exit 1;
  else

    queue.pl --mem 5G \
      $hybrid_dir/log/train_vae_encoded_nnet_${nn_name}.log \
      python3 $nnet_src/$run_script \
      --train_set=$train_set \
      --vae_type=$vae_type \
      --dev_set=$dev_set \
      --batch_size=$batch_size \
      --epochs=$epochs \
      --weight_decay=$weight_decay \
      --num_classes=$num_classes \
      --model_save_interval=$model_save_interval \
      --experiment_name=exp_1 \
      $vae \
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

