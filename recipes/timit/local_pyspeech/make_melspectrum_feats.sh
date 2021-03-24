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
spectrum_type=log
add_noise=clean
add_reverb=clean
fbank_type="mel,1"
write_utt2num_frames=false
check_for_segment="data/train" # Check this data directory for pre-computed segment files

. parse_options.sh || exit 1;

data_dir=$1
feat_dir=$2

echo "$0 $@"

# Convert feat_dir to the absolute file name

feat_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir";} print $dir; ' $feat_dir ${PWD}`

mkdir -p $feat_dir

name=`basename $data_dir`
scp=$data_dir/wav.scp
segment=$data_dir/segments
log_dir=$data_dir/log
mkdir -p $log_dir
log_dir=`realpath ${log_dir}`

if $write_utt2num_frames; then
    add_opts="$add_opts --write_utt2num_frames"
fi

# split files

echo $0": Splitting segment OR scp files for parallalization..."

if [ -f $segment ]; then


  echo $0:"Checking $check_for_segment for precomputed segment files..."
  name_check=`basename $check_for_segment`

  if [ -f $check_for_segment/log/segment_dump/${name_check}_segmentdump.ark ] ; then
    echo $0:"Using segment files from $check_for_segment ..."
    scp_dump_name=$check_for_segment/log/segment_dump/${name_check}_segmentdump
  else
    echo $0:"No segment files found in data directory $check_for_segment ..."
    scp_dump_name=$log_dir/segment_dump/${name}_segmentdump
    echo $0": Dumping Segment files..."
    mkdir -p $log_dir/segment_dump

    scp_dump_name=$log_dir/segment_dump/${name}_segmentdump
    extract-segments scp,p:$scp $segment \
    ark,scp:${scp_dump_name}.ark,${scp_dump_name}.scp
  fi

  echo $0": Splitting Segment files..."

  split_segments=""
  for n in $(seq $nj); do
    split_segments="$split_segments $log_dir/segments.$n"
  done
 utils/split_scp.pl ${scp_dump_name}.scp $split_segments || exit 1;

 echo $0": Computing Mel Spectral features for segment files..."

  # Compute mel spectrum features

  $cmd --mem 5G JOB=1:$nj \
  $log_dir/feats_${name}.JOB.log \
  python ${src_dir}/featgen/computeMelSpectrum.py \
    $log_dir/segments.JOB \
    $feat_dir/melspec_${name}.JOB \
    $add_opts \
    --spectrum_type=$spectrum_type \
    --scp_type="segment" \
    --nfilters=$nfilters \
    --nfft=$nfft \
    --fduration=$fduration \
    --add_noise=$add_noise \
    --add_reverb=$add_reverb \
    --fbank_type=$fbank_type \
    --frate=$frate  || exit 1


    for n in $(seq $nj); do
      cat $feat_dir/melspec_$name.$n.scp || exit 1;
    done > $data_dir/feats.scp

    rm $log_dir/segments.*

    # concatenate all length files together
    if $write_utt2num_frames; then
        for n in $(seq $nj); do
          cat $feat_dir/melspec_$name.$n.len || exit 1;
        done > $data_dir/utt2num_frames
    fi

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
        --spectrum_type=$spectrum_type \
        --nfilters=$nfilters \
        --nfft=$nfft \
        --fduration=$fduration \
        --add_noise=$add_noise \
        --add_reverb=$add_reverb \
        --fbank_type=$fbank_type \
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

echo $0": Finished computing mel spectrum features for $name"
