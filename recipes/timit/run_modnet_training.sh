#!/usr/bin/env bash

mfcc=mfcc
nfilters=23
nj_mfcc=50
nj_melspec=50
nj_modspec=50
nj=50
stage=0
decode_nj=10
train_set=train
test_sets="test dev"
data_dir=data
exp_dir=exp
train=true
align=true
decode=true
tdnn_gmm=tri3
lm="bg"
feat_suffix=""
dataprep=true


# Assume that the data preparation is already done

melspec='melspec'

if [ $stage -le 0 ]; then

  echo "###################################"
  echo "  Mel Spectrum Feature Extraction  "
  echo "###################################"

  for x in ${train_set} ${test_sets} ; do
    local_pyspeech/make_melspectrum_feats.sh --nj $nj_melspec \
      --nfilters $nfilters \
      $data_dir/$x $data_dir/$x/$melspec  || exit 1;
      local_pyspeech/get_cmvn.sh \
      $data_dir/$x $data_dir/$x/$melspec/cmvn || exit 1;
  done
fi

