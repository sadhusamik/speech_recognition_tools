#!/bin/bash

## Train a Hybrid Pytorch Model with best Kaldi alignments
# Samik Sadhu 

stage=0

exp_dir=exp_hybrid
data_dir=data
ali_dir=exp/tri3_ali
mfcc=mfcc

. ./cmd.sh
. ./path.sh
. ./src_paths.sh

. utils/parse_options.sh

if [ $stage -le 0 ]; then 

  echo "## Get alignment for hybrid model ##"
  ./run_get_hq_ali.sh || exit 1;
fi

# mvector parameters

nfilters=20
coeff_0=1
coeff_n=15
fduration=0.5
order=30

modspec=modspec_${nfilters}_${coeff_0}_${coeff_n}_${fduration}_${order}

if [ $stage -le 1 ]; then 
  
  echo "#####################"
  echo "  Extract M-vectors  "
  echo "#####################"

  for x in train test dev ; do
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

if [ $stage -le 2 ]; then 
  
  echo "## Train Hybrid Model in Pytorch ##"

  get_egs=true
  train_nnet=true
  
  hybrid_dir=$exp_dir/hybrid_pytorch
  mkdir -p $hybrid_dir
  
  if $get_egs; then 
    
    for x in train dev; do 
      local_pyspeech/fetch_feats.sh $data_dir/$x $modspec || exit 1;
    done

    cmvn_path=$hybrid_dir/global_cmvn
    compute-cmvn-stats scp:$data_dir/train/feats.scp $cmvn_path  || exit 1;
 
    for x in train dev; do
      egs_dir=$hybrid_dir/egs/$x
      mkdir -p $egs_dir
      python3 $nnet_src/data_prep_feedforward.py \
        --num_chunks=2 \
        --feat_type=cmvn,$cmvn_path \
        --concat_feats=4,4 \
        $data_dir/$x/feats.scp \
        ${ali_dir}_${x} \
        $egs_dir || exit 1;
    done
  fi

  if $train_nnet; then 
    $cuda_cmd --mem 5G \
      $hybrid_dir/log/train_feedforward.log \
      python3 $nnet_src/train_feedforward_nnet.py \
      --use_gpu \
      --train_set=train \
      --dev_sets=dev \
      --num_layers=4 \
      --hidden_dim=512 \
      --batch_size=50000 \
      --epochs=300 \
      --feature_dim=13 \
      --num_frames=9 \
      --num_classes=38 \
      --egs_dir=$hybrid_dir/egs \
      --store_path=$hybrid_dir/nnet_ffwd_4lyrs_512nodes \
      --experiment_name=exp_1 || exit 1;
  fi

fi


if [ $stage -le 3 ] ; then 

  echo "## Extract Posteriors for clean set ##"

  egs_config="$exp_dir/hybrid_pytorch/egs/egs.config"
  model="$exp_dir/hybrid_pytorch/nnet_ffwd_4lyrs_512nodes/exp_1.dir/exp_1__epoch_300.model"
  log_dir=$exp_dir/hybrid_pytorch/log
  mkdir -p $log_dir  
  nj=10

  for x in train dev test; do 
    post_dir=$exp_dir/hybrid_pytorch/post/$x
    mkdir -p $post_dir
    local_pyspeech/fetch_feats.sh $data_dir/$x $mfcc || exit 1;
    split_scp=""
    for n in `seq $nj`; do 
      split_scp="$split_scp $log_dir/${x}.$n.scp"
    done
    utils/split_scp.pl $data_dir/$x/feats.scp $split_scp || exit 1;

    queue.pl JOB=1:$nj \
      $log_dir/${x}_post.JOB.log \
      python3 $nnet_src/extract_posterior.py $model \
      $log_dir/${x}.JOB.scp \
      $egs_config \
      $post_dir/post.JOB || exit 1;
    
    for n in `seq $nj`; do 
      cat $post_dir/post.$n.scp
    done > $post_dir/all_list

  done 

fi

if [ $stage -le 4 ]; then
  
  echo "## STAGE: Noisy MFCC feature extraction ##"
  
  noise_types="babble"
  dbs="5 10 20"
  for noise in $noise_types; do 
    for db in $dbs; do 
      add_opts="--add_noise=${noise},${db}"
      mfcc=mfcc_${noise}_${db}
      for x in test dev; do
        local_pyspeech/make_mfcc_feats.sh --nj 50 \
          --nfilters 23 \
          --add_opts $add_opts \
          $data_dir/$x $data_dir/$x/$mfcc || exit 1;
        local_pyspeech/get_cmvn.sh \
          $data_dir/$x $data_dir/$x/$mfcc/cmvn || exit 1;

      done
    done
  done
fi


if [ $stage -le 5 ] ; then 

  echo "## Extract Posteriors for noisy dev set ##"

  egs_config="$exp_dir/hybrid_pytorch/egs/egs.config"
  model="$exp_dir/hybrid_pytorch/nnet_ffwd_4lyrs_512nodes/exp_1.dir/exp_1__epoch_300.model"
  log_dir=$exp_dir/hybrid_pytorch/log
  mkdir -p $log_dir  
  nj=10
  noise_types="babble"
  dbs="5 10 20"

  for x in dev; do
    for noise in $noise_types; do 
      for db in $dbs; do
        mfcc=mfcc_${noise}_${db}
        post_dir=$exp_dir/hybrid_pytorch/post/${x}_${noise}_${db}
        mkdir -p $post_dir
        local_pyspeech/fetch_feats.sh $data_dir/$x $mfcc || exit 1;
        split_scp=""
        for n in `seq $nj`; do 
          split_scp="$split_scp $log_dir/${x}.$n.scp"
        done
        utils/split_scp.pl $data_dir/$x/feats.scp $split_scp || exit 1;

        queue.pl JOB=1:$nj \
          $log_dir/${x}_post.JOB.log \
          python3 $nnet_src/extract_posterior.py $model \
          $log_dir/${x}.JOB.scp \
          $egs_config \
          $post_dir/post.JOB || exit 1;
        
        for n in `seq $nj`; do 
          cat $post_dir/post.$n.scp
        done > $post_dir/all_list
      done
    done

  done 

fi

if [ $stage -le 6 ]; then 
  
  echo "##  Train RNN AE ##"

  mfcc=mfcc  
  apc_src="../../tools/Autoregressive-Predictive-Coding/"
  rnn_ae_dir=$exp_dir/rnn_ae; mkdir -p $rnn_ae_dir
  post_dir=$exp_dir/hybrid_pytorch/post/
  generate_egs=false
  train_rnn=true
  feat_type="cmvn"

  noise_types="babble"
  dbs="5 10 20"
  bn_dims="20"
  time_steps="0 1 8 20"
  
  if $generate_egs; then
    
    if [ $feat_type == "cmvn" ]; then     
      trans_path=$rnn_ae_dir/global_cmvn
      compute-cmvn-stats \
        scp:$post_dir/train/all_list \
        $trans_path  || exit 1;
    elif [ $feat_type == "pca" ] ; then
      trans_path=$rnn_ae_dir/pca.mat
      shuffle_list.pl $post_dir/train/all_list | sort | \
        est-pca  scp:- $trans_path || exit 1;
    fi
    
    # For all the clean posteriors
    for x in train dev ; do 
      egs_dir=$rnn_ae_dir/egs/${x}
      mkdir -p $egs_dir
      python $apc_src/data_preparation_apc.py \
        --feat_type=${feat_type},${trans_path} \
        --max_seq_len 512 \
        $post_dir/${x} \
        $egs_dir || exit 1;
    done
    
    # For the noisy posteriors

    for x in dev ; do
      for noise in $noise_types; do 
       for db in $dbs; do  
          egs_dir=$rnn_ae_dir/egs/${x}_${noise}_${db}
          mkdir -p $egs_dir
          python $apc_src/data_preparation_apc.py \
            --feat_type=${feat_type},${trans_path} \
            --max_seq_len 512 \
            $post_dir/${x}_${noise}_${db} \
            $egs_dir || exit 1;
        done
      done
    done

    for x in test ; do 
      egs_dir=$rnn_ae_dir/egs/${x}
      mkdir -p $egs_dir
      python $apc_src/data_preparation_apc.py \
        --feat_type=${feat_type},${trans_path} \
        --notruncpad \
        $post_dir/${x} \
        $egs_dir || exit 1;
    done
  fi
  
  for x in test ; do
    for noise in $noise_types; do 
     for db in $dbs; do  
        egs_dir=$rnn_ae_dir/egs/${x}_${noise}_${db}
        mkdir -p $egs_dir
        python $apc_src/data_preparation_apc.py \
          --feat_type=${feat_type},${trans_path} \
          --notruncpad \
          $post_dir/${x}_${noise}_${db} \
          $egs_dir || exit 1;
      done
    done
  done

  if $train_rnn; then 
    egs_dir=$rnn_ae_dir/egs
    for bn_dim in $bn_dims; do 
      for step in $time_steps; do
        
        ( $cuda_cmd --mem 30G \
          $rnn_ae_dir/rnn_ae_bottleNeck${bn_dim}_timeStep${step}/ae_train.log \
          python3 $apc_src/train_rnn_ae.py --egs_dir=$egs_dir \
          --use_gpu \
          --anneal_dev_set=dev \
          --train_set=train \
          --dev_set=dev,dev_babble_5,dev_babble_10,dev_babble_20 \
          --encoder_num_layers=1 \
          --decoder_num_layers=1 \
          --time_shift=$step \
          --loss="MSE" \
          --bn_dim=$bn_dim \
          --feature_dim=38 \
          --epochs=500 \
          --batch_size=64 \
          --store_path=$rnn_ae_dir/rnn_ae_bottleNeck${bn_dim}_timeStep${step} \
          --experiment_name=exp_1 || exit 1 ) &

      done
    done
    wait;
  fi
fi

## Train layerwise RNN AE

if [ $stage -le 7 ] ; then 

  echo "## Extract Posteriors for layerwise RNN-AE  ##"

  egs_config="$exp_dir/hybrid_pytorch/egs/egs.config"
  model="$exp_dir/hybrid_pytorch/nnet_ffwd_4lyrs_512nodes/exp_1.dir/exp_1__epoch_300.model"
  log_dir=$exp_dir/hybrid_pytorch/log

  mkdir -p $log_dir  
  nj=10

  for x in train dev; do
        for layer in 0 1 2; do
          mfcc=mfcc
          post_dir=$exp_dir/hybrid_pytorch/post_layerwise/${x}_layer${layer}
          mkdir -p $post_dir

          local_pyspeech/fetch_feats.sh $data_dir/$x $mfcc || exit 1;
          split_scp=""
          for n in `seq $nj`; do 
            split_scp="$split_scp $log_dir/${x}.$n.scp"
          done
          utils/split_scp.pl $data_dir/$x/feats.scp $split_scp || exit 1;

          queue.pl JOB=1:$nj \
            $log_dir/${x}_post.JOB.log \
            python3 $nnet_src/extract_posterior.py --layer=$layer \
            $model \
            $log_dir/${x}.JOB.scp \
            $egs_config \
            $post_dir/post.JOB || exit 1;
          
          for n in `seq $nj`; do 
            cat $post_dir/post.$n.scp
          done > $post_dir/all_list
      done
  done 
fi

if [ $stage -le 8 ]; then 
  
  echo "##  Train layerwise RNN AE ##"

  mfcc=mfcc  
  apc_src="../../tools/Autoregressive-Predictive-Coding/"
  rnn_ae_dir=$exp_dir/rnn_ae_layerwise_rawfeats; mkdir -p $rnn_ae_dir
  post_dir=$exp_dir/hybrid_pytorch/post_layerwise/
  generate_egs=true
  train_rnn=true
  feat_type="raw"

  bn_dims="20"
  time_steps="0"
  layers="0 1 2"

  if $generate_egs; then
    
    for layer in $layers; do  
      if [ $feat_type == "cmvn" ]; then     
        trans_path=$rnn_ae_dir/global_cmvn_layer${layer}
        compute-cmvn-stats \
          scp:$post_dir/train_layer${layer}/all_list \
          $trans_path  || exit 1;
      elif [ $feat_type == "pca" ] ; then
        trans_path=$rnn_ae_dir/pca_layer${layer}.mat
        shuffle_list.pl $post_dir/train_${layer}/all_list | sort | \
          est-pca  scp:- $trans_path || exit 1;
      fi
    done
    
    for x in train dev ; do
       for layer in $layers; do 
        egs_dir=$rnn_ae_dir/egs/${x}_layer${layer}
        mkdir -p $egs_dir
        trans_path=$rnn_ae_dir/global_cmvn_layer${layer}
        python $apc_src/data_preparation_apc.py \
          --feat_type=raw,None \
          --max_seq_len 512 \
          $post_dir/${x}_layer${layer} \
          $egs_dir || exit 1;
      done
    done

  fi

  if $train_rnn; then 
    egs_dir=$rnn_ae_dir/egs
    for bn_dim in $bn_dims; do 
      for step in $time_steps; do
        for layer in 0; do     
          ( $cuda_cmd --mem 30G \
            $rnn_ae_dir/rnn_ae_bottleNeck${bn_dim}_timeStep${step}_layer${layer}/ae_train.log \
            python3 $apc_src/train_rnn_ae.py --egs_dir=$egs_dir \
            --use_gpu \
            --anneal_dev_set=dev_layer${layer} \
            --train_set=train_layer${layer} \
            --dev_set=dev_layer${layer} \
            --encoder_num_layers=1 \
            --decoder_num_layers=1 \
            --time_shift=$step \
            --loss="MSE" \
            --bn_dim=$bn_dim \
            --feature_dim=38 \
            --epochs=500 \
            --batch_size=64 \
            --terminate_num=20 \
            --store_path=$rnn_ae_dir/rnn_ae_bottleNeck${bn_dim}_timeStep${step}_layer${layer} \
            --experiment_name=exp_1 || exit 1 ) &
        done
      done
    done

    for bn_dim in $bn_dims; do 
      for step in $time_steps; do
        for layer in 1 2 ; do     
          ( $cuda_cmd --mem 30G \
            $rnn_ae_dir/rnn_ae_bottleNeck${bn_dim}_timeStep${step}_layer${layer}/ae_train.log \
            python3 $apc_src/train_rnn_ae.py --egs_dir=$egs_dir \
            --use_gpu \
            --anneal_dev_set=dev_layer${layer} \
            --train_set=train_layer${layer} \
            --dev_set=dev_layer${layer} \
            --encoder_num_layers=1 \
            --decoder_num_layers=1 \
            --time_shift=$step \
            --loss="MSE" \
            --bn_dim=$bn_dim \
            --feature_dim=512 \
            --epochs=500 \
            --batch_size=64 \
            --terminate_num=20 \
            --store_path=$rnn_ae_dir/rnn_ae_bottleNeck${bn_dim}_timeStep${step}_layer${layer} \
            --experiment_name=exp_1 || exit 1 ) &
        done
      done
    done
    wait;
  fi
fi

## Analysis of TIMIT trained model on WSJ

if [ $stage -le 9 ]; then 

  echo "## Extract Posteriors for TIMIT and WSJ test set  ##"

  egs_config="$exp_dir/hybrid_pytorch/egs/egs.config"
  model="$exp_dir/hybrid_pytorch/nnet_ffwd_4lyrs_512nodes/exp_1.dir/exp_1__epoch_300.model"
  log_dir=$exp_dir/hybrid_pytorch/log

  mkdir -p $log_dir  
  nj=10
 <<k 
  # TIMIT TEST SET POSTERIORS
  for x in test; do
        for layer in 0 1 2; do
          mfcc=mfcc
          post_dir=$exp_dir/hybrid_pytorch/post_layerwise/${x}_layer${layer}
          mkdir -p $post_dir

          local_pyspeech/fetch_feats.sh $data_dir/$x $mfcc || exit 1;
          split_scp=""
          for n in `seq $nj`; do 
            split_scp="$split_scp $log_dir/${x}.$n.scp"
          done
          utils/split_scp.pl $data_dir/$x/feats.scp $split_scp || exit 1;

          queue.pl JOB=1:$nj \
            $log_dir/${x}_post.JOB.log \
            python3 $nnet_src/extract_posterior.py --layer=$layer \
            $model \
            $log_dir/${x}.JOB.scp \
            $egs_config \
            $post_dir/post.JOB || exit 1;
          
          for n in `seq $nj`; do 
            cat $post_dir/post.$n.scp
          done > $post_dir/all_list
      done
  done 
k
  # WSJ TEST SET POSTERIORS
  
  for x in test_dev93 ; do
        for layer in 0 1 2; do
          mfcc=mfcc
          post_dir=$exp_dir/hybrid_pytorch/post_layerwise/${x}_layer${layer}
          mkdir -p $post_dir

          local_pyspeech/fetch_feats.sh ../wsj/data/$x $mfcc || exit 1;
          split_scp=""
          for n in `seq $nj`; do 
            split_scp="$split_scp $log_dir/${x}.$n.scp"
          done
          utils/split_scp.pl ../wsj/data/$x/feats.scp $split_scp || exit 1;

          queue.pl JOB=1:$nj \
            $log_dir/${x}_post.JOB.log \
            python3 $nnet_src/extract_posterior.py --layer=$layer \
            $model \
            $log_dir/${x}.JOB.scp \
            $egs_config \
            $post_dir/post.JOB || exit 1;
          
          for n in `seq $nj`; do 
            cat $post_dir/post.$n.scp
          done > $post_dir/all_list
      done
  done 
fi

if [ $stage -le 10 ] ; then 
  
 echo "## RNN AE scores for TIMIT and WSJ test sets  ##"
    
  apc_src="../../tools/Autoregressive-Predictive-Coding/"
  rnn_ae_dir='exp_hybrid/rnn_ae_layerwise'
  
  get_score=true
  generate_egs=true
  
  bn_dim=20
  time_steps="0"
  layers="0"
  score_dir=$exp_dir/pm_scores
  mkdir -p $score_dir

  if $generate_egs ; then

    for layer in $layers; do 
        mfcc=mfcc_${noise}_${db}
        for x in test test_eval92; do 
          egs_dir=$rnn_ae_dir/egs/${x}_layer${layer}
          mkdir -p $egs_dir
              
          python $apc_src/data_preparation_apc.py \
            --notruncpad \
            --feat_type=cmvn,$rnn_ae_dir/global_cmvn_layer${layer} \
            $exp_dir/hybrid_pytorch/post_layerwise/${x}_layer${layer} \
            $egs_dir || exit 1; 
        done
      done
    fi
    
   if $get_score ; then 
      for time_step in $time_steps; do 
        for layer in $layers ; do
          for x in test test_eval92; do 
              (
              queue.pl \
                $exp_dir/hybrid_pytorch/rnn_ae_score_${bn_dim}_${time_step}_layer${layer}_${x}.log\
                python $apc_src/score_utterances_rnn_ae.py \
                 --loss="MSE" \
                 --time_shift=$time_step \
                 $rnn_ae_dir/egs/${x}_layer${layer} \
                 $rnn_ae_dir/rnn_ae_bottleNeck${bn_dim}_timeStep${time_step}_layer${layer}/exp_1.dir/exp_1__epoch_81.model\
                 $score_dir/rnn_ae_${bn_dim}_${time_step}_layer${layer}_${x}.score || exit 1; 
              ) &
          done
        done
      done
      wait
    fi
fi

if [ $stage -le 11 ] ; then 
  
  echo "## TIMIT to WSJ adaptation ##"
  
  adapt_nnet=true
  data_prep=false
  adapt_dir=$exp_dir/adapt_lastlayer_test_dev93
  mkdir -p adapt_dir
  cmvn_path=$exp_dir/hybrid_pytorch/global_cmvn 
 
  if $data_prep; then 

    # WSJ dev set 

    egs_dir=$adapt_dir/egs/test_eval92
    mkdir -p $egs_dir
    python3 $nnet_src/data_prep_feedforward.py \
      --num_chunks=1 \
      --feat_type=cmvn,$cmvn_path \
      --concat_feats=4,4 \
      --ali_type=phone \
      ../wsj/data/test_eval92/feats.scp \
      ../wsj/exp/tri3b_ali_test_eval92/ \
      $egs_dir || exit 1;
    
  fi

  if $adapt_nnet; then 
   
    $cuda_cmd --mem 5G $adapt_dir/adapt.log \
      python3 $nnet_src/nnet_adapt_ae.py \
      --use_gpu \
      --lr_factor=0.8 \
      --store_path=$adapt_dir/take1 \
      --batch_size=100 \
      --max_seq_len=512 \
      --time_shift=0 \
      "exp_hybrid/hybrid_pytorch/nnet_ffwd_4lyrs_512nodes/exp_1.dir/exp_1__epoch_300.model"\
      "exp_hybrid/rnn_ae_layerwise/rnn_ae_bottleNeck20_timeStep0_layer0/exp_1.dir/exp_1__epoch_129.model"\
      ../wsj/data/test_dev93/feats.scp \
      $exp_dir/hybrid_pytorch/egs/egs.config \
      $adapt_dir/egs/test_eval92/chunk_0.pt \
      $exp_dir/rnn_ae_layerwise/global_cmvn_layer0 || exit 1;
  fi
fi

if [ $stage -le 12 ] ; then 
  
  echo "## TIMIT to WSJ adaptation with Regularization ##"
  
  adapt_nnet=true
  data_prep=false
  adapt_dir=$exp_dir/adapt_lastlayer_regu_test_dev93
  mkdir -p adapt_dir
  cmvn_path=$exp_dir/hybrid_pytorch/global_cmvn 
 
  if $data_prep; then 

    # WSJ dev set 

    egs_dir=$adapt_dir/egs/test_eval92
i    mkdir -p $egs_dir
    python3 $nnet_src/data_prep_feedforward.py \
      --num_chunks=1 \
      --feat_type=cmvn,$cmvn_path \
      --concat_feats=4,4 \
      --ali_type=phone \
      ../wsj/data/test_eval92/feats.scp \
      ../wsj/exp/tri3b_ali_test_eval92/ \
      $egs_dir || exit 1;

    egs_dir=$adapt_dir/egs/train
    mkdir -p $egs_dir
    python3 $nnet_src/data_prep_feedforward.py \
      --num_chunks=1 \
      --feat_type=cmvn,$cmvn_path \
      --concat_feats=4,4 \
      --ali_type=phone \
      $data_dir/train/feats.scp \
      exp/tri3_ali_train \
      $egs_dir || exit 1;
    
  fi

  if $adapt_nnet; then 
   
    $cuda_cmd --mem 5G $adapt_dir/adapt.log \
      python3 $nnet_src/nnet_adapt_ae_regularized.py \
      --use_gpu \
      --lr_factor=1 \
      --reg_weight=0.1 \
      --store_path=$adapt_dir/take1 \
      --batch_size=100 \
      --max_seq_len=512 \
      --time_shift=0 \
      "exp_hybrid/hybrid_pytorch/nnet_ffwd_4lyrs_512nodes/exp_1.dir/exp_1__epoch_300.model"\
      "exp_hybrid/rnn_ae_layerwise/rnn_ae_bottleNeck20_timeStep0_layer0/exp_1.dir/exp_1__epoch_129.model"\
      ../wsj/data/test_dev93/feats.scp \
      $exp_dir/hybrid_pytorch/egs/egs.config \
      $adapt_dir/egs/test_eval92/chunk_0.pt \
      $adapt_dir/egs/train/chunk_0.pt \
      $exp_dir/rnn_ae_layerwise/global_cmvn_layer0 || exit 1;
  fi
fi

if [ $stage -le 13 ] ; then 
  
  echo "## TIMIT to WSJ adaptation with Regularization ##"
  
  adapt_nnet=true
  data_prep=true

  reg="0 0.1 0.5 1"

  for r in $reg; do 
    adapt_dir=$exp_dir/adapt_lastlayer_regu_${r}_train_si84
    mkdir -p adapt_dir
    cmvn_path=$exp_dir/hybrid_pytorch/global_cmvn 
   
    if $data_prep; then 

      # WSJ dev set 

      egs_dir=$adapt_dir/egs/test_eval92
      mkdir -p $egs_dir
      python3 $nnet_src/data_prep_feedforward.py \
        --num_chunks=1 \
        --feat_type=cmvn,$cmvn_path \
        --concat_feats=4,4 \
        --ali_type=phone \
        ../wsj/data/test_eval92/feats.scp \
        ../wsj/exp/tri3b_ali_test_eval92/ \
        $egs_dir || exit 1;

      egs_dir=$adapt_dir/egs/train
      mkdir -p $egs_dir
      python3 $nnet_src/data_prep_feedforward.py \
        --num_chunks=1 \
        --feat_type=cmvn,$cmvn_path \
        --concat_feats=4,4 \
        --ali_type=phone \
        $data_dir/train/feats.scp \
        exp/tri3_ali_train \
        $egs_dir || exit 1;
      
    fi

    if $adapt_nnet; then 
     
      $cuda_cmd --mem 5G $adapt_dir/adapt.log \
        python3 $nnet_src/nnet_adapt_ae_regularized.py \
        --use_gpu \
        --lr_factor=1 \
        --learning_rate=0.0001 \
        --reg_weight=$r \
        --store_path=$adapt_dir/take1 \
        --batch_size=100 \
        --max_seq_len=512 \
        --time_shift=0 \
        --epochs=50 \
        "exp_hybrid/hybrid_pytorch/nnet_ffwd_4lyrs_512nodes/exp_1.dir/exp_1__epoch_300.model"\
        "exp_hybrid/rnn_ae_layerwise/rnn_ae_bottleNeck20_timeStep0_layer0/exp_1.dir/exp_1__epoch_129.model"\
        ../wsj/data/train_si84/feats.scp \
        $exp_dir/hybrid_pytorch/egs/egs.config \
        $adapt_dir/egs/test_eval92/chunk_0.pt \
        $adapt_dir/egs/train/chunk_0.pt \
        $exp_dir/rnn_ae_layerwise/global_cmvn_layer0 || exit 1;
    fi
  done
fi
exit
if [ $stage -le 14 ] ; then 
  
  echo "## TIMIT to WSJ adaptation with Delta RNN-AE  ##"
  
  adapt_nnet=true
  data_prep=false

  adapt_dir=$exp_dir/adapt_lastlayer_delta_train_si84
  mkdir -p adapt_dir
  cmvn_path=$exp_dir/hybrid_pytorch/global_cmvn 
 
  if $data_prep; then 

    # WSJ dev set 

    egs_dir=$adapt_dir/egs/test_eval92
    mkdir -p $egs_dir
    python3 $nnet_src/data_prep_feedforward.py \
      --num_chunks=1 \
      --feat_type=cmvn,$cmvn_path \
      --concat_feats=4,4 \
      --ali_type=phone \
      ../wsj/data/test_eval92/feats.scp \
      ../wsj/exp/tri3b_ali_test_eval92/ \
      $egs_dir || exit 1;

    egs_dir=$adapt_dir/egs/train
    mkdir -p $egs_dir
    python3 $nnet_src/data_prep_feedforward.py \
      --num_chunks=1 \
      --feat_type=cmvn,$cmvn_path \
      --concat_feats=4,4 \
      --ali_type=phone \
      $data_dir/train/feats.scp \
      exp/tri3_ali_train \
      $egs_dir || exit 1;
    
  fi

  if $adapt_nnet; then 
   
    $cuda_cmd --mem 5G $adapt_dir/adapt.log \
      python3 $nnet_src/nnet_adapt_ae_delta.py \
      --use_gpu \
      --lr_factor=1 \
      --learning_rate=0.0001 \
      --store_path=$adapt_dir/take1 \
      --batch_size=100 \
      --max_seq_len=512 \
      --time_shift=20 \
      --epochs=50 \
      "exp_hybrid/hybrid_pytorch/nnet_ffwd_4lyrs_512nodes/exp_1.dir/exp_1__epoch_300.model"\
      "exp_hybrid/rnn_ae_layerwise/rnn_ae_bottleNeck20_timeStep0_layer0/exp_1.dir/exp_1__epoch_129.model"\
      "exp_hybrid/rnn_ae/rnn_ae_bottleNeck20_timeStep20/exp_1.dir/exp_1__epoch_17.model"\
      ../wsj/data/train_si84/feats.scp \
      $exp_dir/hybrid_pytorch/egs/egs.config \
      $adapt_dir/egs/test_eval92/chunk_0.pt \
      $exp_dir/rnn_ae_layerwise/global_cmvn_layer0 || exit 1;
  fi
fi
exit
if [ $stage -le 12 ] ; then 
  
  echo "## TIMIT to WSJ adaptation with layerwise RNN-AE##"
  
  adapt_nnet=true
  data_prep=false
  adapt_dir=$exp_dir/adapt_multilayer_test_dev93
  mkdir -p adapt_dir
  cmvn_path=$exp_dir/hybrid_pytorch/global_cmvn 
 
  if $data_prep; then 

    # WSJ test set 
    egs_dir=$adapt_dir/egs/test_eval92
    mkdir -p $egs_dir
    python3 $nnet_src/data_prep_feedforward.py \
      --num_chunks=1 \
      --feat_type=cmvn,$cmvn_path \
      --concat_feats=4,4 \
      --ali_type=phone \
      ../wsj/data/test_eval92/feats.scp \
      ../wsj/exp/tri3b_ali_test_eval92/ \
      $egs_dir || exit 1;
    
  fi

  if $adapt_nnet; then 
    layers="0 1 2"
    model="exp_hybrid/hybrid_pytorch/nnet_ffwd_4lyrs_512nodes/exp_1.dir/exp_1__epoch_300.model" 
    com="exp_hybrid/rnn_ae_layerwise/global_cmvn_layer"
    cmvn="${com}0,${com}1,${com}2"

    com="exp_hybrid/rnn_ae_layerwise/rnn_ae_bottleNeck20_timeStep0_layer"
    pm_models="${com}0/exp_1.dir/exp_1__epoch_129.model,${com}1/exp_1.dir/exp_1__epoch_500.model,${com}2/exp_1.dir/exp_1__epoch_500.model"
    $cuda_cmd --mem 5G $adapt_dir/adapt.log \
      python3 $nnet_src/nnet_adapt_ae_multilayer.py \
      --use_gpu \
      --lr_factor=1 \
      --store_path=$adapt_dir/take1 \
      --batch_size=64 \
      --max_seq_len=512 \
      --time_shift=0 \
      $model \
      $pm_models \
      ../wsj/data/test_dev93/feats.scp \
      $exp_dir/hybrid_pytorch/egs/egs.config \
      $adapt_dir/egs/test_eval92/chunk_0.pt \
      $cmvn || exit 1;
  fi
fi


