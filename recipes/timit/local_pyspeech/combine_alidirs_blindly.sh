#!/bin/bash 


ali_dir_new=$1
shift;
first_dir=$1

mkdir -p $ali_dir_new

for dir in $*; do 
  counter=1
  echo "$0: Copying ali files from $dir to $ali_dir_new"
  data_name=`echo $dir | rev | cut -d"/" -f 2 | rev`
 for alifile in `ls $dir/ali*gz`; do 
    cp -r $alifile $ali_dir_new/ali.$data_name.$counter.gz
    counter=$((counter+1))
 done
done

cp $first_dir/final.mdl $ali_dir_new/final.mdl
