#!/bin/bash 
# Samik Sadhu 

cmd=queue.pl
nj=20
tandem_type=presoftmax
get_pca=false
stage=0
feat_type=raw
cmvn_opts=

. utils/parse_options.sh

echo "$0 $@"

nnet_dir=$1
data_dir=$2
data_name=`basename $data_dir`
sdata=$data_dir/split${nj}
tandem_dir=$3

log_dir=$tandem_dir/log
post_dir=$tandem_dir/post/$data_name
mkdir -p $log_dir $post_dir $post_dir/$tandem_type
post_dir=`realpath $post_dir`
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`

. ./path.sh

case $feat_type in 
  raw) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |";;
  plain) feats="ark:copy-feats scp:$sdata/JOB/feats.scp ark:- |";; 
  splice_lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- |"
    feats="$feats transform-feats $nnet_dir/final.mat ark:- ark:- |";;
  nosplice_lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
    feats="$feats transform-feats $nnet_dir/final.mat ark:- ark:- |";;
  *) echo "$0: Invalid feature type $feat_type" 
esac

echo "$0: Using feature type $feat_type"

if [ ! -f $tandem_dir/tandem.mdl ]; then
  case $tandem_type in 
    presoftmax) comp_num=`nnet-am-info $nnet_dir/final.mdl | grep num-components | cut -d' ' -f2`
      nnet-to-raw-nnet $nnet_dir/final.mdl - | \
        raw-nnet-copy --truncate=$((comp_num-1)) - $tandem_dir/tandem.mdl || exit1;;
    
    softmax) nnet-to-raw-nnet $nnet_dir/final.mdl $tandem_dir/tandem.mdl;;

    *) echo "$0: Tandem type $tandem_type is not valid"
  esac
else
  echo "$0: tandem.mdl already found in $tandem_dir"
fi

if [ $stage -le 0 ]; then
  split_data.sh $data_dir $nj || exit 1;

  $cmd JOB=1:$nj $log_dir/forward_pass_${data_name}.JOB.log \
    nnet-compute --pad-input \
    $tandem_dir/tandem.mdl \
    "$feats" \
    ark,scp:$post_dir/$tandem_type/post.JOB.ark,$post_dir/$tandem_type/post.JOB.scp || exit 1;

  for n in `seq $nj`; do 
    cat $post_dir/$tandem_type/post.$n.scp 
  done > $post_dir/post.scp 
    
fi

if [ $stage -le 1 ]; then 
  if $get_pca; then 
    shuffle_list.pl $post_dir/post.scp | sort | \
      est-pca  scp:- $tandem_dir/post/pca.mat || exit 1;
  fi
fi



