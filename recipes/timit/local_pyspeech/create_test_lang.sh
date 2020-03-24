#!/bin/bash

# This script formats ARPA LM into G.fst.

lm_dir=$1
lang_dir=$2
dir=$3

arpa_lm=$lm_dir/3gram-mincount/lm_unpruned.gz
#dir=data/lang_test

if [ -f ./path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh

[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

mkdir -p $dir
cp -r $lang_dir/* $dir

gunzip -c "$arpa_lm" | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$dir/words.txt - $dir/G.fst


echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic $dir/G.fst

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=$lang_dir/phones.txt --osymbols=$lang_dir/words.txt \
  $lang_dir/L.fst  | head

echo Performing further checks

# Checking that G.fst is determinizable.
fstdeterminize $dir/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize $dir/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
fsttablecompose $dir/L_disambig.fst $dir/G.fst | \
   fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose $lang_dir/L_disambig.fst $dir/G.fst | \
   fstisstochastic || echo "[log:] LG is not stochastic"

echo "$0 succeeded"
