#!/bin/bash 


# Code to simplify the dictionary created for WSJ by Kaldi to TIMIT set of
# phonemes
# Samik Sadhu
#


dict=data/local/dict_nosp

phone_map=$1

while read line; do 
  num=`echo $line | awk '{print NF}'`;
  if [[ $num -ge 2 ]]; then 
    base=`echo $line | cut -d' ' -f1` ; 
    for seg in $line; do  
      sed -i -- "s/$seg/$base/g" $dict/lexicon.txt;
      sed -i -- "s/$seg/$base/g" $dict/lexicon2_raw.txt;
      sed -i -- "s/$seg/$base/g" $dict/lexicon1_raw_nosil.txt;

    done  
  fi 
done < $phone_map

cp $dict/lexicon.txt $dict/temp.txt
awk '!a[$0]++' $dict/temp.txt > $dict/lexicon.txt
cp $dict/lexicon2_raw.txt $dict/temp.txt
awk '!a[$0]++' $dict/temp.txt > $dict/lexicon2_raw.txt
cp $dict/lexicon1_raw_nosil.txt $dict/temp.txt
awk '!a[$0]++' $dict/temp.txt > $dict/lexicon1_raw_nosil.txt

rm $dict/temp.txt
rm $dict/extra_questions.txt
touch $dict/extra_questions.txt

rm $dict/nonsilence_phones.txt
cat $phone_map | cut -d' ' -f1 | head -37 | sort > $dict/nonsilence_phones.txt
rm $dict/silence_phones.txt
cat $phone_map | cut -d' ' -f1 | tail -1 > $dict/silence_phones.txt
