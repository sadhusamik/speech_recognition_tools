#/bin/bash 

wsj=data/local/dict_nosp
librispeech=../librispeech/data/local/dict_nosp
fisher_english=../fisher_english/data/local/dict


dict_new=data/local/dict_universal
mkdir -p $dict_new 
# Copy general files
for file in extra_questions.txt optional_silence.txt nonsilence_phones.txt silence_phones.txt; do 
  cp $wsj/$file $dict_new/$file || exit 1;
done
# Merge lexicon & lexiconp files

tr '[:lower:]' '[:upper:]' < $fisher_english/lexiconp.txt | \
  sed -e 's/\t/  /g' | tr -s " " > $dict_new/fish.temp
sed -e 's/\t/  /g' $wsj/lexiconp.txt | tr -s " " > $dict_new/wsj.temp
sed -e 's/\t/  /g' $librispeech/lexiconp.txt | tr -s " " > $dict_new/libri.temp

sort $dict_new/wsj.temp $dict_new/libri.temp $dict_new/fish.temp | \
  uniq > $dict_new/lexiconp.txt

tr '[:lower:]' '[:upper:]' < $fisher_english/lexicon.txt | \
  sed -e 's/\t/  /g' | tr -s " " > $dict_new/fish.temp
sed -e 's/\t/  /g' $wsj/lexicon.txt | tr -s " " > $dict_new/wsj.temp
sed -e 's/\t/  /g' $librispeech/lexicon.txt | tr -s " " > $dict_new/libri.temp

sort $dict_new/wsj.temp $dict_new/libri.temp $dict_new/fish.temp | \
  uniq > $dict_new/lexicon.txt

rm $dict_new/{fish.temp,wsj.temp,libri.temp}
