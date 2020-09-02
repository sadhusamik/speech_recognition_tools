#!/bin/bash

## Get high quality alignments for WSJ data
## Samik Sadhu 

stage=0
exp_dir=exp
data_dir=data
simplify_dict=true # Simplify dictionary to 37 phonemes + 1 silence

train=true
decode=false
align=true

. ./cmd.sh

. utils/parse_options.sh

wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B

. ./path.sh

mfcc=mfcc
nj=50

if [ $stage -le 0 ]; then

  echo "## STAGE: Data Prepartion ##"
  local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;
  local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;
  if $simplify_dict; then 
    echo "SIMPLIFYING DICTIONARY"
    local_pyspeech/simplify_dictionary.sh conf/phone_map_wsj
  fi
  utils/prepare_lang.sh  --position_dependent_phones false \
    data/local/dict_nosp \
    "<SPOKEN_NOISE>" \
    data/local/lang_tmp_nosp \
    data/lang_nosp || exit 1;
  local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;
  
fi

for x in silence_phones.txt nonsilence_phones.txt; do 
  cat data/local/dict_nosp/$x
done > conf/phones.txt

if [ $stage -le 1 ]; then
  
  echo "## STAGE: MFCC feature extraction ##"

  for x in test_eval92 test_dev93 train_si284; do
    local_pyspeech/make_mfcc_feats.sh --nj $nj \
      --nfilters 23 \
      data/$x data/$x/$mfcc $add_opts || exit 1;
    local_pyspeech/get_cmvn.sh \
      data/$x data/$x/$mfcc/cmvn || exit 1;
  done
  
  utils/subset_data_dir.sh --first data/train_si284 7138 data/train_si84 || exit 1
  utils/subset_data_dir.sh --shortest data/train_si84 2000 data/train_si84_2kshort || exit 1;
  utils/subset_data_dir.sh data/train_si84 3500 data/train_si84_half || exit 1;

fi

if [ $stage -le 2 ]; then 

  for x in test_eval92 test_eval93 test_dev93 train_si284; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 data/$x || exit 1;
    steps/compute_cmvn_stats.sh data/$x || exit 1;  done


  utils/subset_data_dir.sh --first data/train_si284 7138 data/train_si84 || exit 1
  utils/subset_data_dir.sh --shortest data/train_si84 2000 data/train_si84_2kshort || exit 1;
  utils/subset_data_dir.sh data/train_si84 3500 data/train_si84_half || exit 1;
fi


if [ $stage -le 3 ]; then
  # monophone

  if $train; then
    steps/train_mono.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
      data/train_si84_2kshort data/lang_nosp exp/mono0a || exit 1;
  fi
  decode=true
  if $decode; then
    utils/mkgraph.sh data/lang_nosp_test_tgpr exp/mono0a exp/mono0a/graph_nosp_tgpr && \
      steps/decode.sh --nj 10 --cmd "$decode_cmd" exp/mono0a/graph_nosp_tgpr \
        data/test_dev93 exp/mono0a/decode_nosp_tgpr_dev93 && \
      steps/decode.sh --nj 8 --cmd "$decode_cmd" exp/mono0a/graph_nosp_tgpr \
        data/test_eval92 exp/mono0a/decode_nosp_tgpr_eval92
  fi
  decode=false
  if $align; then 
    
    for x in  train_si284 test_dev93 test_eval92; do
      steps/align_si.sh --boost-silence 1.25 --nj 8 --cmd "$train_cmd" \
        data/$x data/lang_nosp exp/mono0a exp/mono0a_ali_$x || exit 1;
    done
  fi
fi

if [ $stage -le 3 ]; then
  # tri1
  if $train; then
    steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
      data/train_si84_half data/lang_nosp exp/mono0a exp/mono0a_ali || exit 1;

    steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
      data/train_si84_half data/lang_nosp exp/mono0a_ali exp/tri1 || exit 1;
  fi

  if $decode; then
    utils/mkgraph.sh data/lang_nosp_test_tgpr \
      exp/tri1 exp/tri1/graph_nosp_tgpr || exit 1;

    for data in dev93 eval92; do
      nspk=$(wc -l <data/test_${data}/spk2utt)
      steps/decode.sh --nj $nspk --cmd "$decode_cmd" exp/tri1/graph_nosp_tgpr \
        data/test_${data} exp/tri1/decode_nosp_tgpr_${data} || exit 1;

      # test various modes of lm rescoring (4 is the default one).
      # this is just confirming they're equivalent.
      for mode in 1 2 3 4 5; do
        steps/lmrescore.sh --mode $mode --cmd "$decode_cmd" \
          data/lang_nosp_test_{tgpr,tg} data/test_${data} \
          exp/tri1/decode_nosp_tgpr_${data} \
          exp/tri1/decode_nosp_tgpr_${data}_tg$mode  || exit 1;
      done
      # later on we'll demonstrate const-arpa lm rescoring, which is now
      # the recommended method.
    done

  fi
fi

if [ $stage -le 4 ]; then
  # tri2b.  there is no special meaning in the "b"-- it's historical.
  if $train; then
    steps/align_si.sh --nj 10 --cmd "$train_cmd" \
      data/train_si84 data/lang_nosp exp/tri1 exp/tri1_ali_si84 || exit 1;

    steps/train_lda_mllt.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
      data/train_si84 data/lang_nosp exp/tri1_ali_si84 exp/tri2b || exit 1;
  fi

  if $decode; then
    utils/mkgraph.sh data/lang_nosp_test_tgpr \
      exp/tri2b exp/tri2b/graph_nosp_tgpr || exit 1;
    for data in dev93 eval92; do
      nspk=$(wc -l <data/test_${data}/spk2utt)
      steps/decode.sh --nj ${nspk} --cmd "$decode_cmd" exp/tri2b/graph_nosp_tgpr \
        data/test_${data} exp/tri2b/decode_nosp_tgpr_${data} || exit 1;

       # compare lattice rescoring with biglm decoding, going from tgpr to tg.
      steps/decode_biglm.sh --nj ${nspk} --cmd "$decode_cmd" \
        exp/tri2b/graph_nosp_tgpr data/lang_nosp_test_{tgpr,tg}/G.fst \
        data/test_${data} exp/tri2b/decode_nosp_tgpr_${data}_tg_biglm

       # baseline via LM rescoring of lattices.
      steps/lmrescore.sh --cmd "$decode_cmd" \
        data/lang_nosp_test_tgpr/ data/lang_nosp_test_tg/ \
        data/test_${data} exp/tri2b/decode_nosp_tgpr_${data} \
        exp/tri2b/decode_nosp_tgpr_${data}_tg || exit 1;

      # Demonstrating Minimum Bayes Risk decoding (like Confusion Network decoding):
      mkdir exp/tri2b/decode_nosp_tgpr_${data}_tg_mbr
      cp exp/tri2b/decode_nosp_tgpr_${data}_tg/lat.*.gz \
         exp/tri2b/decode_nosp_tgpr_${data}_tg_mbr;
      local/score_mbr.sh --cmd "$decode_cmd"  \
         data/test_${data}/ data/lang_nosp_test_tgpr/ \
         exp/tri2b/decode_nosp_tgpr_${data}_tg_mbr
    done
  fi

  # At this point, you could run the example scripts that show how VTLN works.
  # We haven't included this in the default recipes.
  # local/run_vtln.sh --lang-suffix "_nosp"
  # local/run_vtln2.sh --lang-suffix "_nosp"
fi


if [ $stage -le 5 ]; then
  # From 2b system, train 3b which is LDA + MLLT + SAT.

  # Align tri2b system with all the si284 data.
  if $train; then
    steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
      data/train_si284 data/lang_nosp exp/tri2b exp/tri2b_ali_si284  || exit 1;

    steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
      data/train_si284 data/lang_nosp exp/tri2b_ali_si284 exp/tri3b || exit 1;
  fi
  decode=true
  if $decode; then
    utils/mkgraph.sh data/lang_nosp_test_tgpr \
      exp/tri3b exp/tri3b/graph_nosp_tgpr || exit 1;

    for data in dev93 eval92; do
      nspk=$(wc -l <data/test_${data}/spk2utt)
    
      steps/decode_fmllr.sh --nj ${nspk} --cmd "$decode_cmd" \
        exp/tri3b/graph_nosp_tgpr data/test_${data} \
        exp/tri3b/decode_nosp_tgpr_${data} || exit 1;
    done
  fi
  
  # Finally get alignments for train test and dev sets 

  for x in train_si284 test_dev93 test_eval92 ; do 
    steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" \
      data/$x data/lang_nosp exp/tri3b exp/tri3b_ali_${x} || exit 1;
  done
fi

