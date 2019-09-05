#!/bin/bash 


mfcc=mfcc
nfilters=23
nj_mfcc=50
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

. ./path.sh
. ./cmd.sh

. utils/parse_options.sh

if $dataprep ; then 

  timit=/export/corpora5/LDC/LDC93S1/timit/TIMIT # @JHU
  #timit=/mnt/matylda2/data/TIMIT/timit # @BUT

  local/timit_data_prep.sh $timit || exit 1

  local/timit_prepare_dict.sh

  # Caution below: we remove optional silence by setting "--sil-prob 0.0",
  # in TIMIT the silence appears also as a word in the dictionary and is scored.
  utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 \
   data/local/dict "sil" data/local/lang_tmp data/lang

  local/timit_format_data.sh
  if [ $data_dir != "data" ]; then
    mv data $data_dir
  fi
fi


if [ $stage -le 0 ]; then 

  echo "###########################"
  echo "  MFCC Feature Extraction  "
  echo "###########################"
 
  for x in ${train_set} ${test_sets} ; do
    local_pyspeech/make_mfcc_feats.sh --nj $nj_mfcc \
      --nfilters $nfilters \
      $data_dir/$x $data_dir/$x/$mfcc  || exit 1;
      local_pyspeech/get_cmvn.sh \
      $data_dir/$x $data_dir/$x/$mfcc/cmvn || exit 1;
  done
fi


if [ $stage -le 1 ]; then
  
  echo "#####################################"
  echo "  Monophone HMM-GMM Model with MFCC  "
  echo "#####################################"

  if $train; then

    # Starting basic training on MFCC features
    steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
            $data_dir/${train_set} $data_dir/lang $exp_dir/mono_$mfcc
    fi

  if $decode ; then
    for x in ${test_sets} ; do 
      local_pyspeech/fetch_feats.sh --feat_source pyspeech \
        $data_dir/$x $mfcc || exit 1;
    done

    utils/mkgraph.sh $data_dir/lang_test_$lm $exp_dir/mono_$mfcc $exp_dir/mono_$mfcc/graph
    for dset in ${test_sets}; do
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
          $exp_dir/mono_$mfcc/graph $data_dir/${dset} $exp_dir/mono_$mfcc/decode_${dset} &
    done
    wait
  fi
fi

if [ $stage -le 2 ]; then

  echo "#####################################"
  echo "  Triphone HMM-GMM Model with MFCC  "
  echo "#####################################"

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		    $data_dir/${train_set} \
        $data_dir/lang \
        $exp_dir/mono_$mfcc \
        $exp_dir/mono_${mfcc}_ali || exit 1

  steps/train_deltas.sh --cmd "$train_cmd" \
			2500 15000 $data_dir/${train_set} \
      $data_dir/lang \
      $exp_dir/mono_${mfcc}_ali \
      $exp_dir/tri1_${mfcc} || exit 1
fi

if [ $stage -le 3 ]; then

  echo "#####################################"
  echo "  LDA-MLLT HMM-GMM Model with MFCC  "
  echo "#####################################"
  
  if $align ; then
    steps/align_si.sh --nj $nj --cmd "$train_cmd" \
          $data_dir/${train_set} \
          $data_dir/lang \
          $exp_dir/tri1_$mfcc \
          $exp_dir/tri1_${mfcc}_ali || exit 1
  fi
  
  if $train; then
    steps/train_lda_mllt.sh --cmd "$train_cmd" \
          2500 15000 $data_dir/${train_set} \
          $data_dir/lang \
          $exp_dir/tri1_${mfcc}_ali \
          $exp_dir/tri2_${mfcc} || exit 1
  fi

  if $decode ; then

    utils/mkgraph.sh $data_dir/lang_test_$lm \
      $exp_dir/tri2_$mfcc \
      $exp_dir/tri2_$mfcc/graph

    for dset in ${test_sets}; do
      (
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
          $exp_dir/tri2_$mfcc/graph \
          $data_dir/${dset} \
          $exp_dir/tri2_$mfcc/decode_${dset}${feat_suffix} 
      ) &
  done
  wait

  fi
fi


if [ $stage -le 4 ]; then 

  echo "###############################  "
  echo "  Triphone SAT system with MFCC  "
  echo "###############################  "
   
  if $align; then 
   
    for x in ${train_set}; do  
      local_pyspeech/fetch_feats.sh --feat_source pyspeech \
        $data_dir/$x \
        $mfcc || exit 1;
    done

    steps/align_si.sh --nj 80 --cmd "$train_cmd" \
      $data_dir/$train_set \
      $data_dir/lang \
      $exp_dir/tri2_${mfcc} \
      $exp_dir/tri2_${mfcc}_ali || exit 1;
  
  fi
  
  if $train ; then

   for x in ${train_set}; do  
      local_pyspeech/fetch_feats.sh --feat_source pyspeech \
        $data_dir/$x \
        $mfcc || exit 1;
    done

    steps/train_sat.sh --cmd "$train_cmd" \
      2500 15000 \
      $data_dir/${train_set} \
      $data_dir/lang \
      $exp_dir/tri2_${mfcc}_ali \
      $exp_dir/tri3_${mfcc}  || exit 1;

  fi

  if $decode ; then

    
    utils/mkgraph.sh $data_dir/lang_test_$lm \
      $exp_dir/tri3_${mfcc} \
      $exp_dir/tri3_${mfcc}/graph 

    for dset in ${test_sets}; do
      steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
          $exp_dir/tri3_${mfcc}/graph \
          $data_dir/${dset} \
          $exp_dir/tri3_${mfcc}/decode_${dset}${feat_suffix} &
    done
    wait

  fi

  if $align; then 
      
    steps/align_fmllr.sh --nj 80 --cmd "$train_cmd" \
      $data_dir/$train_set \
      $data_dir/lang \
      $exp_dir/tri3_${mfcc} \
      $exp_dir/tri3_${mfcc}_ali || exit 1;
  fi

fi


# mvector parameters

nfilters=23
coeff_0=1
coeff_n=15
fduration=0.5
order=30

modspec=modspec_${nfilters}_${coeff_0}_${coeff_n}_${fduration}_${order}

if [ $stage -le 5 ]; then 
  
  echo "#####################"
  echo "  Extract M-vectors  "
  echo "#####################"

  for x in ${train_set} ${test_sets} ; do
    local_pyspeech/make_modspec_feats.sh --nj $nj_modspec \
      --nfilters $nfilters \
      --coeff_0 $coeff_0 \
      --coeff_n $coeff_n \
      --fduration $fduration \
      --order $order \
      $data_dir/$x \
      $data_dir/$x/$modspec || exit 1;

    local_pyspeech/get_cmvn.sh \
      $data_dir/$x $data_dir/$x/$modspec/cmvn || exit 1;

  done

fi

if [ $stage -le 6 ]; then 
  
  echo "#########################################################"
  echo "  Triphone Hybrid Model with M-vectors and SAT Alignment "
  echo "#########################################################"
     
  if $align; then 
   
    for x in ${train_set}; do  
      local_pyspeech/fetch_feats.sh --feat_source pyspeech \
        $data_dir/$x \
        $mfcc || exit 1;
    done

    steps/align_fmllr.sh --nj 80 --cmd "$train_cmd" \
      $data_dir/$train_set \
      $data_dir/lang \
      $exp_dir/tri3_${mfcc} \
      $exp_dir/tri3_${mfcc}_ali || exit 1;
  
  fi

  if $train ; then

   for x in ${train_set}; do  
      local_pyspeech/fetch_feats.sh --feat_source pyspeech \
        $data_dir/$x \
        $modspec || exit 1;
    done
    
    steps/nnet2/train_tanh.sh --cmd "queue.pl --gpu 1" \
     --feat_type "raw" \
     --hidden_layer_dim 256 \
     --num_hidden_layers 4 \
     --num_epochs 20 \
     --stage -20 \
     --splice-width 4 \
     --num_jobs_nnet 20 \
     --num_threads 1 \
     --parallel_opts "--num-threads 1 --mem 1G" \
     $data_dir/$train_set \
     $data_dir/lang \
     $exp_dir/tri3_${mfcc}_ali \
     $exp_dir/hybrid_tri3_${modspec} || exit 1;
  fi

  if $decode; then  
    for x in ${test_sets}; do
      (
      local_pyspeech/fetch_feats.sh --feat_source pyspeech \
        $data_dir/$x \
        $modspec || exit 1;

      [ ! -d $exp_dir/hybrid_tri3_$modspec/decode_$x ] && mkdir \
        $exp_dir/hybrid_tri3_$modspec/decode_$x
    
      steps/nnet2/decode.sh --cmd "$decode_cmd" \
        --feat_type "raw" \
        --nj 20 \
        --num-threads 6 \
        $exp_dir/tri3_${mfcc}/graph \
        $data_dir/$x \
        $exp_dir/hybrid_tri3_${modspec}/decode_$x |\
        tee $exp_dir/hybrid_tri3_${modspec}/decode_$x/decode.log ;
      ) & 
  done
  wait
  fi

fi
