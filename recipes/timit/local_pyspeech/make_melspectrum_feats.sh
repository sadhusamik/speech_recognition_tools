#!/usr/bin/env bash

. ./path.sh

# Mel Spectrum options

nj=100
nfilters=15
context=
nfft=1024
fduration=0.02
frate=100
cmd=queue.pl
add_opts=
src_dir='../../src'

. parse_options.sh || exit 1;

data_dir=$1
feat_dir=$2

echo "$0 $@"

# Convert feat_dir to the absolute file name

feat_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir";} print $dir; ' $feat_dir ${PWD}`

mkdir -p $feat_dir

name=`basename $data_dir`
scp=$data_dir/wav.scp
segment=$data_dir/segment
log_dir=$data_dir/log
mkdir -p $log_dir

# split files

echo $0": Splitting segment OR scp files for parallalization..."

if [ -f $segment ]; then

 ## TODO: THIS PART IS NOT IMPLEMENTED YET! Do not extract features for segment files

  echo $0": Splitting Segment files..."

  split_segments=""
  for n in $(seq $nj); do
    split_segments="$split_segments $log_dir/segments.$n"
  done
 utils/split_scp.pl $segment $split_segments || exit 1;

 echo $0": Computing Mel Spectral features for segment files..."

  # Compute mfcc features

  $cmd --mem 5G JOB=1:$nj \
    $log_dir/feats_${name}.JOB.log \
    python computeMelSpectrum.py \
      $log_dir/wav_${name}.JOB.scp \
      $feat_dir/mfcc_${name}.JOB \
      $add_opts \
      --nfilters=$nfilters \
      --nfft=$nfft \
      --fduration=$fduration \
      --frate=$frate \
      --kaldi_cmd=$ark_cmd || exit 1

    # concatenate all scp files together

    for n in $(seq $nj); do
      cat $feat_dir/melspec_$name.$n.scp || exit 1;
    done > $data_dir/feats.scp

    rm $log_dir/wav_${name}.*.scp

elif [ -f $scp ]; then

  echo "$0: Splitting scp files..."

  split_scp=""
  for n in $(seq $nj); do
    split_scp="$split_scp $log_dir/wav_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scp || exit 1;

  echo "$0: Computing Mel Spectral features for scp files..."

  # Compute mel spectrum features

    $cmd --mem 5G JOB=1:$nj \
      $log_dir/feats_${name}.JOB.log \
      python ${src_dir}/featgen/computeMelSpectrum.py \
        $log_dir/wav_${name}.JOB.scp \
        $feat_dir/melspec_${name}.JOB \
        $add_opts \
        --nfilters=$nfilters \
        --nfft=$nfft \
        --fduration=$fduration \
        --frate=$frate  || exit 1

    # concatenate all scp files together

    for n in $(seq $nj); do
      cat $feat_dir/melspec_$name.$n.scp || exit 1;
    done > $data_dir/feats.scp

    rm $log_dir/wav_${name}.*.scp


else
  echo "$0: Neither scp file nor segment file exists... something is wrong!"
  exit 1;
fi


echo $0": Finished computed mfcc features for $name"
