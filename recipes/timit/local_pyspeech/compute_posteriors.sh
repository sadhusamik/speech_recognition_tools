#!/usr/bin/env bash

. ./path.sh
. ./src_paths.sh 

stage=0
nj=10
hybrid_dir=exp/hybrid_generative_pytorch
data_dir=data
feat_type=mfcc
nn_name=nnet_gru_3lenc_1lclas_1lae_256nodes
pw=0.2
num_threads=8
model_iter=
model_override=
append=
override_cmvn=
override_model=
ae_type=normal
test_set=test

. utils/parse_options.sh

post_dir=$hybrid_dir/posterior_$append/$test_set/post
mkdir -p $post_dir
log_dir=$hybrid_dir/log
mkdir -p $log_dir

if [ $stage -le 0 ]; then 
  echo "$0: Compute Posteriors"

  for x in $test_set ; do 
    local_pyspeech/fetch_feats.sh $data_dir/$x $feat_type || exit 1;
  done

  split_scp=""
  for n in `seq $nj`; do 
    split_scp="$split_scp $log_dir/${test_set}.$n.scp"
  done
  utils/split_scp.pl $data_dir/$test_set/feats.scp $split_scp || exit 1;

  if [ -z $model_iter ] ; then
    echo "$0: Choosing best model"
    model="$hybrid_dir/$nn_name/exp_1.dir/final.mdl"
  else
    echo "$0: Choosing model at epoch $model_iter"
    model="$hybrid_dir/$nn_name/exp_1.dir/exp_1__epoch_${model_iter}.model"
   fi
  if [ -z $override_model ]; then 
    echo "$0: Using model $model"
  else
    model=$override_model
    echo "$0: Overriding with model $model"
  fi

  if [ -z $override_cmvn ]; then 
    add_opts=""
  else
    echo "$0: Overriding cmvn with given file"
    add_opts="--override_trans=$override_cmvn"
  fi

  queue.pl JOB=1:$nj \
    $log_dir/compute_posteriors.JOB.log \
    python3 $nnet_src/dump_genclassifier_outputs.py $add_opts \
    --ae_type=$ae_type \
    $model \
    $log_dir/${test_set}.JOB.scp \
    $hybrid_dir/egs/egs.config \
    $post_dir/post_${test_set}.JOB || exit 1;
   
  for n in `seq $nj`; do
   cat $post_dir/post_${test_set}.$n.scp 
  done > $hybrid_dir/posterior_$append/$test_set/feats.scp
fi
