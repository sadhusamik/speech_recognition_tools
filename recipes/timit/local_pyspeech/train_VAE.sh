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
concat_train_set=train
dev_set=dev
concat_dev_set=dev
nn_name=nnet_gru_3lenc_1lclas_1lae_256nodes
num_egs_jobs=2
egs_dir=
concat_egs_dir=

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
use_transformer=false

# Feature config
feature_dim=
left_context=
right_context=
max_seq_len=512
ali_type="phone"
ali_append=
per_utt_cmvn=true
skip_cmvn=false
do_pca=false
out_dist='gauss'

. utils/parse_options.sh || exit 1;

if [ -z ${feature_dim} ] && [ -z ${egs_dir} ]; then
  feature_dim=`feat-to-dim scp:${data_dir}/${train_set}/feats.scp -`
fi

if [ -z ${egs_dir} ] ; then
    if [ -z ${feature_dim} ] ; then echo "Set feature_dim when providing egs_dir"; exit 1; fi
fi

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
  normal_AE)
    echo "$0: Train simple VAE"
    vae_script=train_VAE.py
    add_vae_opts="--only_AE --encoder_num_layers=$encoder_num_layers --decoder_num_layers=$decoder_num_layers --hidden_dim=$hidden_dim --bn_dim=$bn_dim"
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

if [ $stage -le 0 ] && [ -z ${egs_dir} ]; then

  if $skip_cmvn; then
    echo "$0: No cmvn computed..."
  elif $per_utt_cmvn; then
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

  if $skip_cmvn && $do_pca; then
    echo "$0: Computing a PCA transform for the training data"
    utils/shuffle_list.pl $data_dir/${train_set}/feats.scp | sort |\
      est-pca scp:- $hybrid_dir/pca_all.mat
  fi

  for x in $train_set $dev_set $test_set ; do
    egs_dir=$hybrid_dir/egs/$x
    mkdir -p $egs_dir

    if $skip_cmvn; then
      cmvn_opts=""
    elif $per_utt_cmvn; then
      cmvn_path=$hybrid_dir/perutt_cmvn_${x}_${feat_type}.scp
      cmvn_opts="--feat_type=$cmvn_type,$cmvn_path"
    else
      cmvn_path=$hybrid_dir/global_cmvn_${feat_type}
      cmvn_opts="--feat_type=$cmvn_type,$cmvn_path"
    fi

    if $skip_cmvn && $do_pca; then
      cmvn_opts="--feat_type=pca,$hybrid_dir/pca_all.mat"
    fi

    if [ ! -z ${left_context} ] && [ ! -z ${right_context} ] ; then
      cmvn_opts+=" --concat_feats=${left_context},${right_context}"
    fi

    if [ ! -z $ali_append ]; then 
      ali_name=${ali_append}_${x}
    else
      ali_name=$x
    fi
    
    python3 $nnet_src/data_prep_for_seq.py $cmvn_opts\
      --num_jobs=$num_egs_jobs \
      --ali_type=$ali_type \
      --max_seq_len=$max_seq_len \
      $data_dir/$x/feats.scp \
      ${hmm_dir}_ali_${ali_name} \
      $egs_dir || exit 1;
  done

fi

if [ -z ${egs_dir} ]; then
  egs_dir=$hybrid_dir/egs
fi

if [ ! -z ${concat_egs_dir} ]; then
  add_vae_opts="$add_vae_opts --concat_egs_dir=${concat_egs_dir} --concat_train_set=${concat_train_set} --concat_dev_set=${concat_dev_set}"
fi

if ${use_transformer}; then
   add_vae_opts="$add_vae_opts --use_transformer"
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
      --out_dist=$out_dist \
      --model_save_interval=$model_save_interval \
      --experiment_name=exp_1 \
      $egs_dir \
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
      --out_dist=$out_dist \
      --model_save_interval=$model_save_interval \
      --experiment_name=exp_1 \
      $egs_dir \
      $hybrid_dir/$nn_name || exit 1;
  fi

fi

