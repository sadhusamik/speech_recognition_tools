#!/usr/bin/env bash

. ./path.sh
. ./src_paths.sh 

stage=0
nj=10
hybrid_dir=exp/hybrid_generative_pytorch
data_dir=data
lang_dir=data/lang_test_bg
graph_name=graph
feat_type=mfcc
hmm_dir=exp/tri3
test_set=test
train_set=train
nn_name=nnet_gru_3lenc_1lclas_1lae_256nodes
pw=0.2
num_threads=8
model_iter=
model_override=
append=
score_script=score.sh
override_cmvn=
override_model=
override_egs_config=
override_priors=
compute_cmvn=false
override_prior=

# Decoder parameters
min_active=200
max_active=700
beam=8
lattice_beam=13
acwt=0.2
remove_ll=true # Remove loglikelihood directory after decoding 

. utils/parse_options.sh 

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

ll_dir=$hybrid_dir/loglikelihoods_${test_set}_${feat_type}_${append}
decode_dir=$hybrid_dir/decode_${test_set}_${feat_type}_${append}
mkdir -p $ll_dir
mkdir -p $hybrid_dir
log_dir=$hybrid_dir/log
mkdir -p $log_dir
mkdir -p $decode_dir


if [ $stage -le 0 ]; then 
  echo "$0: Compute Log-likelihood"

  cmvn_path=$hybrid_dir/perutt_cmvn_${test_set}_${feat_type}
  compute-cmvn-stats \
    scp:$data_dir/$test_set/feats.scp \
    ark,scp:$cmvn_path.ark,$cmvn_path.scp  || exit 1;
  
  add_opts="--override_trans=cmvn_utt,$cmvn_path.scp"
  
  if [ ! -z $override_egs_config ]; then
    echo "$0: Overriding egs.config file" 
    egs_config_file=$override_egs_config
  else
    egs_config_file=$hybrid_dir/egs/${train_set}/egs.config
  fi

  if [ ! -z $override_priors ]; then
    echo "$0: Overriding prior file" 
    prior_file=$override_priors
  else
    prior_file=$hybrid_dir/priors
  fi

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

  queue.pl --mem 10G JOB=1:$nj \
    $log_dir/compute_llikelihood_${test_set}.JOB.log \
    python3 $nnet_src/compute_CURL_classifier_likelihood.py $add_opts \
    --prior=$prior_file \
    --prior_weight=$pw \
    $model \
    $log_dir/${test_set}.JOB.scp \
    $hybrid_dir/egs/${train_set}/egs.config \
    $ll_dir/$test_set.JOB.ll || exit 1;
   
  for n in `seq $nj`; do
   cat $ll_dir/$test_set.$n.ll.scp 
  done > $ll_dir/all_llhoods
fi

if [ $stage -le 1 ]; then 
  echo "$0: Make graph and Decode " 
  utils/mkgraph.sh \
    $lang_dir $hmm_dir $hmm_dir/$graph_name || exit 1;

  split_scp=""
  for n in `seq $nj`; do 
    split_scp="$split_scp $log_dir/${test_set}_ll.$n.scp"
  done
  utils/split_scp.pl $ll_dir/all_llhoods $split_scp || exit 1;

  queue.pl --mem 2G --num-threads $num_threads JOB=1:$nj \
    $log_dir/decode_${test_set}.JOB.log \
    latgen-faster-mapped$thread_string --min-active=$min_active \
    --max-active=$max_active \
    --beam=$beam \
    --lattice-beam=$lattice_beam \
    --acoustic-scale=$acwt \
    --allow-partial=true \
    --word-symbol-table=$hmm_dir/$graph_name/words.txt \
    $hmm_dir/final.mdl \
    $hmm_dir/$graph_name/HCLG.fst \
    scp:$log_dir/${test_set}_ll.JOB.scp \
    "ark:|gzip -c > $decode_dir/lat.JOB.gz" || exit 1;

  echo $nj > $decode_dir/num_jobs
fi

if [ $stage -le 2 ]; then 
  echo "$0: get WER "

  local_pyspeech/$score_script \
    --cmd "queue.pl" \
    --min-lmwt 1 \
    --max-lmwt 10 \
    $data_dir/$test_set \
    $hmm_dir/$graph_name \
    $decode_dir || exit 1;
fi

if $remove_ll; then
  echo "$0: Removing all files from log-likelihood directory"
  rm -r $ll_dir
fi
