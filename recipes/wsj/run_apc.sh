#!/bin/bash

## Train APC Models for surfae features on WSJ
# Samik Sadhu 

stage=0

exp_dir=exp
data_dir=data

train=true
decode=true
align=false
simplify_dict=false

. ./cmd.sh

. utils/parse_options.sh

wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B

. ./path.sh

melspec=melspec_white_10dB
nj=50

if [ $stage -le 0 ]; then

  echo "## STAGE: Data Prepartion ##"
  
  local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;

  local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;

  utils/prepare_lang.sh  --position_dependent_phones false \
    data/local/dict_nosp \
    "<SPOKEN_NOISE>" \
    data/local/lang_tmp_nosp \
    data/lang_nosp || exit 1;

  local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;

  cp -r data $data_dir
fi

for x in silence_phones.txt nonsilence_phones.txt; do 
  cat $data_dir/data/local/dict_nosp/$x
done > conf/phones.txt

if [ $stage -le 1 ]; then 
  
  echo "## STAGE: MELSPEC feature extraction ##"
  
 # add_opts='--add_reverb=large_room' 
  for x in test_eval92 test_dev93 train_si284; do
    local_pyspeech/make_melspectrum_feats.sh --nj $nj \
      --nfilters 80 \
      --add_opts "--add_noise=white,0" \
      $data_dir/$x $data_dir/$x/$melspec  || exit 1;

    local_pyspeech/get_cmvn.sh \
      $data_dir/$x $data_dir/$x/$melspec/cmvn || exit 1;

  done

fi

apc_src="../../tools/Autoregressive-Predictive-Coding/"
time_shift=5

if [ $stage -le 2 ]; then 
  
  echo "## STAGE: Train APC Model ##"
 
  for x in test_eval92 test_dev93 train_si284; do 
    
    egs_dir=$exp_dir/apc_$melspec/egs/$x
    mkdir -p $egs_dir

    local_pyspeech/fetch_feats.sh $data_dir/$x $melspec || exit 1;
    
    python $apc_src/data_preparation_apc.py $data_dir/$x/$melspec \
      $egs_dir || exit 1; 

  done

  egs_dir=$exp_dir/apc_$melspec/egs
  $cuda_cmd $exp_dir/apc_${melspec}/time_shift_${time_shift}/apc_train.log python $apc_src/train_apc.py --egs_dir=$egs_dir \
    --use_gpu \
    --time_shift=$time_shift \
    --store_path=$exp_dir/apc_${melspec}/time_shift_${time_shift} \
    --experiment_name=exp_1 || exit 1

  
fi


if [ $stage -le 3 ]; then 

  echo "## STAGE: Do domain inference ##"
  
  inf_dir=$exp_dir/apc_inf/
  mkdir -p $inf_dir

  queue.pl $inf_dir/apc_inf.log python $apc_src/simple_domain_inference.py \
    "exp/apc_melspec_babble_20dB/egs/test_eval92/,exp/apc_melspec_white_10dB/egs/test_eval92/"\
    "exp/apc_melspec_babble_20dB/time_shift_5/exp_1.dir/exp_1__epoch_99.model,exp/apc_melspec_white_20dB/time_shift_5/exp_1.dir/exp_1__epoch_99.model"\
    $exp_dir/apc_inf/sample.inf \
    --time_shift=$time_shift || exit 1;


fi
