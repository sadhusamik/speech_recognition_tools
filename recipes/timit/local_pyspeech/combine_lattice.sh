#!/bin/bash

num_jobs=20

. utils/parse_options.sh

exp_dir=$1 
decode_src_dirs=$2
decode_dest_dir=$3

mkdir -p $decode_dest_dir
log_dir=$exp_dir/log
mkdir -p $log_dir

. ./path.sh
. ./cmd.sh

src_lats=""
for src in $decode_src_dirs; do 
  src_lats="$src_lats 'ark:gunzip -c $exp_dir/$src/lat.JOB.gz|'"
done

queue.pl --mem 2G JOB=1:$num_jobs \
  $log_dir/decode_combinelats.JOB.log \
  lattice-combine $src_lats \
  "ark:|gzip -c > $decode_dest_dir/lat.JOB.gz" || exit 1;
