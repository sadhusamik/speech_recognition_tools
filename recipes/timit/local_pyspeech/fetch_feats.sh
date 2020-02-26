#!/bin/bash

## Script to fetch the selected features from a data dir into feats.scp and
## cmvn.scp
## Samik Sadhu 

cmvn=true
feat_source=pyspeech #pyspeech or kaldi

. ./utils/parse_options.sh

echo "$0 $@"

data_dir=$1
feat_type=$2
name=`basename $data_dir`
root_feat=`echo $feat_type | cut -f1 -d'_'`
basefeat=$feat_type

counter=1
while true ; do

  case $feat_source in 
    kaldi) fname=mfcc/raw_mfcc_$name.$counter.scp
            echo $fname ;;
    pyspeech) fname=$data_dir/$feat_type/${root_feat}_$name.$counter.scp ;;
    *) echo "Feature type $feat_source is nor recognized!"
       exit 1;;
  esac 

  if [ -f $fname ] ; then 
    cat $fname; 
    counter=$((counter+1))
  else
    break;
  fi

done > $data_dir/feats.scp

if $cmvn; then
  case $feat_source in 
    kaldi)  cmvn_dir=mfcc ;;
    pyspeech) cmvn_dir=$data_dir/$feat_type/cmvn;;
  esac
  
  if [ -f $cmvn_dir/cmvn_$name.scp ]; then
    cp $cmvn_dir/cmvn_$name.scp $data_dir/cmvn.scp || exit 1;
  else
    echo "$0: Cant find cmvn skipping.."
  fi
fi

echo $0": Compiled all the precomputed $feat_type features from $data_dir !"
