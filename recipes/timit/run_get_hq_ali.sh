#!/bin/bash 

## Get high quality phonetic alignments for TIMT data


. ./cmd.sh
[ -f path.sh ] && . ./path.sh
set -e

# Acoustic model parameters
numLeavesTri1=2500
numGaussTri1=15000
numLeavesMLLT=2500
numGaussMLLT=15000
numLeavesSAT=2500
numGaussSAT=15000
numGaussUBM=400
numLeavesSGMM=7000
numGaussSGMM=9000

feats_nj=10
train_nj=30
decode_nj=10
stage=5
train=false
decode=false

if [ $stage -le 0 ]; then 

  timit=/export/corpora5/LDC/LDC93S1/timit/TIMIT # @JHU

  local/timit_data_prep.sh $timit || exit 1
  local/timit_prepare_dict.sh
  utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 \
   data/local/dict "sil" data/local/lang_tmp data/lang
  local/timit_format_data.sh

fi

if [ $stage -le 1 ]; then
  
  echo "## STAGE: MFCC feature extraction ##"

  for x in train test dev; do
    local_pyspeech/make_mfcc_feats.sh --nj 50 \
      --nfilters 23 \
      data/$x data/$x/$mfcc $add_opts || exit 1;
    local_pyspeech/get_cmvn.sh \
      data/$x data/$x/$mfcc/cmvn || exit 1;
  done
fi


echo ============================================================================
echo "                     MonoPhone Training & Decoding                        "
echo ============================================================================
if [ $stage -le 2 ]; then
  if $train; then 
    steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" data/train data/lang exp/mono
  fi

  if $decode; then
    utils/mkgraph.sh data/lang_test_bg exp/mono exp/mono/graph

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/mono/graph data/dev exp/mono/decode_dev

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/mono/graph data/test exp/mono/decode_test
  fi
fi
echo ============================================================================
echo "           tri1 : Deltas + Delta-Deltas Training & Decoding               "
echo ============================================================================

if [ $stage -le 3 ]; then
  if $train; then
    steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
     data/train data/lang exp/mono exp/mono_ali

    # Train tri1, which is deltas + delta-deltas, on train data.
    steps/train_deltas.sh --cmd "$train_cmd" \
     $numLeavesTri1 $numGaussTri1 data/train data/lang exp/mono_ali exp/tri1
  fi

  if $decode; then
    utils/mkgraph.sh data/lang_test_bg exp/tri1 exp/tri1/graph

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri1/graph data/dev exp/tri1/decode_dev

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri1/graph data/test exp/tri1/decode_test
  fi
fi
echo ============================================================================
echo "                 tri2 : LDA + MLLT Training & Decoding                    "
echo ============================================================================
if [ $stage -le 4 ]; then 
  if $train; then 
    steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
      data/train data/lang exp/tri1 exp/tri1_ali

    steps/train_lda_mllt.sh --cmd "$train_cmd" \
     --splice-opts "--left-context=3 --right-context=3" \
     $numLeavesMLLT $numGaussMLLT data/train data/lang exp/tri1_ali exp/tri2
  fi

  if $decode; then 
    utils/mkgraph.sh data/lang_test_bg exp/tri2 exp/tri2/graph

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri2/graph data/dev exp/tri2/decode_dev

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri2/graph data/test exp/tri2/decode_test
  fi
fi
echo ============================================================================
echo "              tri3 : LDA + MLLT + SAT Training & Decoding                 "
echo ============================================================================
if [ $stage -le 5 ]; then 
  if $train; then
    # Align tri2 system with train data.
    steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
     --use-graphs true data/train data/lang exp/tri2 exp/tri2_ali

    # From tri2 system, train tri3 which is LDA + MLLT + SAT.
    steps/train_sat.sh --cmd "$train_cmd" \
     $numLeavesSAT $numGaussSAT data/train data/lang exp/tri2_ali exp/tri3
  fi
#  decode=true
  if $decode; then
    utils/mkgraph.sh data/lang_test_bg exp/tri3 exp/tri3/graph

    steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri3/graph data/dev exp/tri3/decode_dev

    steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri3/graph data/test exp/tri3/decode_test
  fi


  for x in train test dev ; do
    steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
     data/$x data/lang exp/tri3 exp/tri3_ali_${x}
  done
fi


exit
echo ============================================================================
echo "                        SGMM2 Training & Decoding                         "
echo ============================================================================
if [ $stage -le 6 ]; then
  if $train; then
    steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
     data/train data/lang exp/tri3 exp/tri3_ali

    steps/train_ubm.sh --cmd "$train_cmd" \
     $numGaussUBM data/train data/lang exp/tri3_ali exp/ubm4

    steps/train_sgmm2.sh --cmd "$train_cmd" $numLeavesSGMM $numGaussSGMM \
     data/train data/lang exp/tri3_ali exp/ubm4/final.ubm exp/sgmm2_4
  fi
 # decode=true
  if $decode; then
    utils/mkgraph.sh data/lang_test_bg exp/sgmm2_4 exp/sgmm2_4/graph

    steps/decode_sgmm2.sh --nj "$decode_nj" --cmd "$decode_cmd"\
     --transform-dir exp/tri3/decode_dev exp/sgmm2_4/graph data/dev \
     exp/sgmm2_4/decode_dev

    steps/decode_sgmm2.sh --nj "$decode_nj" --cmd "$decode_cmd"\
     --transform-dir exp/tri3/decode_test exp/sgmm2_4/graph data/test \
     exp/sgmm2_4/decode_test
  fi
  # Finally get all the alignments 
  for x in train test dev ; do
    steps/align_sgmm2.sh --nj 20 --cmd "$train_cmd" \
     --transform-dir exp/tri3_ali --use-graphs true --use-gselect true \
     data/$x data/lang exp/sgmm2_4 exp/sgmm2_4_ali_${x}
  done
fi

