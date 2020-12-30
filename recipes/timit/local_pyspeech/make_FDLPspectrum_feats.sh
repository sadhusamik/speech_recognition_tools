#!/usr/bin/env bash

. ./path.sh

# Mel Spectrum options

nj=100
nfilters=20
fduration=0.5
coeff_num=50
coeff_range='0,30'
order=50
overlap_fraction=0.25
add_reverb=clean
add_noise=clean
fbank_type="mel,1"
gamma_weight="None"

frate=100
cmd=queue.pl
add_opts=
src_dir='../../src'
spectrum_type=log
write_utt2num_frames=false

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

if $write_utt2num_frames; then
    add_opts="$add_opts --write_utt2num_frames"
fi

# split files

echo $0": Splitting segment OR scp files for parallalization..."

if [ -f $segment ]; then

    ## TODO: THIS PART IS NOT IMPLEMENTED YET! Do not extract features for segment files
    echo "Not Implemented"
elif [ -f $scp ]; then

  echo "$0: Splitting scp files..."

  split_scp=""
  for n in $(seq $nj); do
    split_scp="$split_scp $log_dir/wav_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scp || exit 1;

  echo "$0: Computing FDLP Spectral features for scp files..."

  # Compute mel spectrum features

    $cmd --mem 5G JOB=1:$nj \
      $log_dir/feats_${name}.JOB.log \
      python ${src_dir}/featgen/computeFDLPSpectrogram.py \
        $log_dir/wav_${name}.JOB.scp \
        $feat_dir/melspec_${name}.JOB \
        $add_opts \
        --fbank_type=$fbank_type \
        --gamma_weight=$gamma_weight \
        --add_reverb=$add_reverb \
        --add_noise=$add_noise \
        --coeff_num=$coeff_num \
        --coeff_range=$coeff_range \
        --order=$order \
        --overlap_fraction=$overlap_fraction \
        --nfilters=$nfilters \
        --fduration=$fduration \
        --frate=$frate  || exit 1

    # concatenate all scp files together

    for n in $(seq $nj); do
      cat $feat_dir/melspec_$name.$n.scp || exit 1;
    done > $data_dir/feats.scp

    rm $log_dir/wav_${name}.*.scp

    # concatenate all length files together
    if $write_utt2num_frames; then
        for n in $(seq $nj); do
          cat $feat_dir/melspec_$name.$n.len || exit 1;
        done > $data_dir/utt2num_frames
    fi


else
  echo "$0: Neither scp file nor segment file exists... something is wrong!"
  exit 1;
fi


echo $0": Finished computed mfcc features for $name"
