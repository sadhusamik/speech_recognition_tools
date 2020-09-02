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
encoder_num_layers=2
decoder_num_layers=2
classifier_num_layers=1
in_channels=1,32,64
out_channels=32,64,128
kernel=3,5
ae_num_layers=1
hidden_dim=256
bn_dim=100
batch_size=64
epochs=300
model_save_interval=10
weight_decay=0.001
vae_type=modulation
nopool=true
beta=1
nfilters=6
nrepeats=50
ar_steps=3,5
filt_type=ellip
reg_weight=0.1
bn_bits=16

# Feature config
feature_dim=13
left_context=6
right_context=6
max_seq_len=512
ali_type="phone"
ali_append=
per_utt_cmvn=true

. utils/parse_options.sh || exit 1;

mkdir -p $hybrid_dir 
log_dir=$hybrid_dir/log

case $vae_type in
  ae_regularized) 
    echo "$0: Training AE Regularized"
    vae_script=train_AE_modulation_regularized.py
    add_vae_opts="--in_channels=$in_channels --out_channels=$out_channels --kernel=$kernel --bn_dim=$bn_dim --reg_weight=$reg_weight"
    ;;
  cnn_nopool_regularized) 
    echo "$0: Training CNN Nopool VAE"
    vae_script=train_CNN_VAE_modulation_reg.py
    add_vae_opts="--in_channels=$in_channels --out_channels=$out_channels --kernel=$kernel --bn_dim=$bn_dim --reg_weight=$reg_weight --bn_bits=$bn_bits"
    ;;
  rs_cnn)
    echo "$0: Training Rate Scale CNN VAE"
    vae_script=train_rsCNN_VAE.py
    add_vae_opts="--in_channels=$in_channels --out_channels=$out_channels --kernel=$kernel --bn_dim=$bn_dim --beta=$beta"
    ;;
  cnn_nopool) 
    echo "$0: Training CNN Nopool VAE"
    vae_script=train_CNN_VAE_nopool.py
    add_vae_opts="--in_channels=$in_channels --out_channels=$out_channels --kernel=$kernel --bn_dim=$bn_dim"
    ;;
  cnn)
    echo "$0: Training CNN VAE"
    vae_script=train_CNN_VAE.py
    add_vae_opts="--in_channels=$in_channels --out_channels=$out_channels --kernel=$kernel --bn_dim=$bn_dim"
    ;;
  cnn_modulation)
    vae_script=train_modulation_CNN_VAE.py
    add_vae_opts="--beta=$beta --in_channels=$in_channels --out_channels=$out_channels --kernel=$kernel --nfilters=$nfilters --nrepeats=$nrepeats --filt_type=$filt_type"
    ;;
  cnn_modulation_nopool) 
    echo "$0: Training CNN Nopool Modulation VAE"
    vae_script=train_modulation_CNN_VAE_nopool.py
    add_vae_opts="--in_channels=$in_channels --out_channels=$out_channels --kernel=$kernel --bn_bits=$bn_bits --nfilters=$nfilters --nrepeats=$nrepeats --filt_type=$filt_type "
    ;;
  modulation)
    echo "$0: Training Modulation VAE"
    vae_script=train_modulation_VAE.py
    add_vae_opts="--beta=$beta --nfilters=$nfilters --nrepeats=$nrepeats --filt_type=$filt_type"
    ;;
  normal)
    echo "$0: Train simple VAE"
    vae_script=train_VAE.py
    add_vae_opts="--encoder_num_layers=$encoder_num_layers --decoder_num_layers=$decoder_num_layers --hidden_dim=$hidden_dim --bn_dim=$bn_dim"
    ;;
  normal_post)
    echo "$0: Train simple VAE"
    vae_script=train_posterior_VAE.py
    add_vae_opts="--encoder_num_layers=$encoder_num_layers --decoder_num_layers=$decoder_num_layers --hidden_dim=$hidden_dim --bn_dim=$bn_dim --bn_bits=$bn_bits"
    ;;
  arvae)
    echo "$0: Train AR VAE"
    vae_script=train_ARVAE.py
    add_vae_opts="--bn_dim=$bn_dim --ar_steps=$ar_steps"
    ;;
  *)
    echo "$0: No VAE type by $vae_type!"
    exit 1;
    ;;
esac

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

  for x in $train_set $dev_set $test_set ; do
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

if [ $stage -le 1 ]; then 
  if $use_gpu; then 
    $cuda_cmd --mem 5G \
      $hybrid_dir/log/train_VAE_${nn_name}.log \
      python3 $nnet_src/$vae_script $add_vae_opts \
      --use_gpu \
      --train_set=$train_set \
      --dev_set=$dev_set \
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
      $hybrid_dir/log/train_VAE_${nn_name}.log \
      python3 $nnet_src/$vae_script $add_vae_opts\
      --train_set=$train_set \
      --dev_set=$dev_set \
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

