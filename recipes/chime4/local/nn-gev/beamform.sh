#!/usr/bin/env bash

for flist in tr05_simu tr05_real dt05_simu dt05_real et05_simu et05_real; do
     $HOME/miniconda3/bin/python local/nn-gev/beamform.py $flist "$@"
done
