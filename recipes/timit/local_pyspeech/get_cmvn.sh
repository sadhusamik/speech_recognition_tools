#!/bin/bash 

. ./path.sh 

per_utt=false

. utils/parse_options.sh

data_dir=$1
cmvn_dir=$2
log_dir=$data_dir/log

# make cmvndir an absolute pathname 
cmvn_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir";} print $dir; ' $cmvn_dir ${PWD}`

name=`basename $data_dir`

mkdir -p $cmvn_dir $log_dir

if $per_utt; then 

! compute-cmvn-stats \
  scp,p:$data_dir/feats.scp \
  ark,scp:$cmvn_dir/cmvn_$name.ark,$cmvn_dir/cmvn_$name.scp \
  2> $log_dir/cmvn_$name.log \
  && echo "Error computing CMVN stats. See $log_dir/cmvn_$name.log" \
  && exit 1;

else

! compute-cmvn-stats \
  --spk2utt=ark:$data_dir/spk2utt \
  scp,p:$data_dir/feats.scp \
  ark,scp:$cmvn_dir/cmvn_$name.ark,$cmvn_dir/cmvn_$name.scp \
  2> $log_dir/cmvn_$name.log \
  && echo "Error computing CMVN stats. See $log_dir/cmvn_$name.log" \
  && exit 1;

fi

cp $cmvn_dir/cmvn_$name.scp $data_dir/cmvn.scp || exit 1;

echo "Finished computing CMVN stats for $name"
