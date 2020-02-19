#!/bin/bash 

model_dir_old=$1
model_dir_new=$2
ali_dir_old=$3
ali_dir_new=$4

mkdir -p $ali_dir_new

for x in `ls $ali_dir_old/ali.*.gz`; do 
  ali_name=`echo $x | rev | cut -d'/' -f1 | rev`
  convert-ali $model_dir_old/final.mdl \
    $model_dir_new/final.mdl \
    $model_dir_new/tree \
    "ark:gunzip -c $x |" \
    "ark:|gzip -c > $ali_dir_new/$ali_name"
  cp $model_dir_new/final.mdl $ali_dir_new/final.mdl || exit 1;
done
