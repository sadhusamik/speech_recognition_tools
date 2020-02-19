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
nn_name=nnet_gru_3lenc_1lclas_1lae_256nodes
pw=0.2
num_threads=8
model=
append=
score_script=score.sh
override_cmvn=

# Decoder parameters
min_active=200
max_active=700
beam=8
lattice_beam=13
acwt=0.2

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

  for x in $test_set ; do 
    local_pyspeech/fetch_feats.sh $data_dir/$x $feat_type || exit 1;
  done

  split_scp=""
  for n in `seq $nj`; do 
    split_scp="$split_scp $log_dir/${test_set}.$n.scp"
  done
  utils/split_scp.pl $data_dir/$test_set/feats.scp $split_scp || exit 1;
  if [ -z $model ] ; then
    echo "$0: Choosing best model"
     model="$hybrid_dir/$nn_name/exp_1.dir/final.mdl"
   fi
  
  if [ -z $override_cmvn ]; then 
    add_opts=""
  else
    echo "$0: Overriding cmvn with given file"
    add_opts="--override_cmvn=$override_cmvn"
  fi

  queue.pl JOB=1:$nj \
    $log_dir/compute_llikelihood.JOB.log \
    python3 $nnet_src/dump_genclassifier_outputs.py $add_opts \
    --prior=$hybrid_dir/priors \
    --prior_weight=$pw \
    $model \
    $log_dir/${test_set}.JOB.scp \
    $hybrid_dir/egs/egs.config \
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
    $log_dir/decode_${teste_set}.JOB.log \
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
