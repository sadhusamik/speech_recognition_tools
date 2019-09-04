#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 12:44:06 2018

@author: samiksadhu
"""

'Computing FDLP Modulation Spectral Features with segment files from Kaldi' 

import argparse
import io
import numpy as np
import os
from scipy.io.wavfile import read
import subprocess
import scipy.fftpack as freqAnalysis 
import sys
import time 

from features import getFrames,createFbank,computeLpcFast,computeModSpecFromLpc, addReverb, dict2Ark

    
def getFeats(args, srate=16000, window=np.hanning):
    
    wavs=args.scp
    segment=args.segment
    outfile=args.outfile
    add_reverb=args.add_reverb
    set_unity_gain=args.set_unity_gain
    nmodulations=args.nmodulations
    order=args.order
    fduration=args.fduration
    frate=args.frate
    nfilters=args.nfilters
    kaldi_cmd=args.kaldi_cmd
    
    fbank = createFbank(nfilters, int(2*fduration*srate), srate)
    
    if add_reverb:
        if add_reverb=='small_room':
            sr_r, rir=read('./RIR/RIR_SmallRoom1_near_AnglA.wav')
            rir=rir[:,1]
            rir=rir/np.power(2,15)
        elif add_reverb=='large_room':
            sr_r, rir=read('./RIR/RIR_LargeRoom1_far_AnglA.wav')
            rir=rir[:,1]
            rir=rir/np.power(2,15)
        elif add_reverb=='clean':
            print('%s: No reverberation added!' % sys.argv[0])
        else:
            raise ValueError('Invalid type of reverberation!')
            
    wav_in_buffer='' # Wav file that is currently in RAM
    
    # Load Location and Ids of all wav files 
    wav_ids=[]; wav_locs=[]
    with open(wavs, 'r') as fid:
        for line in fid:
             tokens = line.strip().split()
             uttid, inwav = tokens[0], ' '.join(tokens[1:])
             wav_ids.append(uttid)
             wav_locs.append(inwav)
    
    # Compute features for all the segments
    with open(segment, 'r') as fid_s:
         all_feats={}
         for line_s in fid_s:
            token_s = line_s.strip().split()
            seg_id=token_s[0]; wav_id=token_s[1] 
            
            # Load wav file it is already not in RAM unload
            if wav_in_buffer!=wav_id:
               wav_in_buffer=wav_id
               inwav=wav_locs[wav_ids.index(wav_id)]
               if inwav[-1] == '|':
                   proc = subprocess.run(inwav[:-1], shell=True,
                                         stdout=subprocess.PIPE)
                   sr, signal_big = read(io.BytesIO(proc.stdout))
               else:
                   sr, signal_big = read(inwav)
                   assert sr == srate, 'Input file has different sampling rate.'
               
            t_beg=int(float(token_s[2])*sr); t_end=int(float(token_s[3])*sr)
            signal=signal_big[t_beg:t_end]
            signal=signal/np.power(2,15)

            if add_reverb:
                if not add_reverb=='clean':
                    signal=addReverb(signal,rir)
                
            time_frames = np.array([frame for frame in
                getFrames(signal, srate, frate, fduration, window)])

            cos_trans=freqAnalysis.dct(time_frames)/np.sqrt(2*int(srate * fduration))
            
            [frame_num, ndct]=np.shape(cos_trans)
                            
            feats=np.zeros((frame_num,nfilters*nmodulations))
            print('%s: Computing Features for file: %s and segment: %s' % (sys.argv[0],wav_id,seg_id))
            sys.stdout.flush()
            for i in range(frame_num):
                each_feat=np.zeros([nfilters,nmodulations])
                for j in range(nfilters):
                    filt=fbank[j,0:-1]
                    band_dct=filt*cos_trans[i,:]
                    xlpc, gg=computeLpcFast(band_dct,order) # Compute LPC coefficients 
                    if set_unity_gain:
                        gg=1
                    mod_spec=computeModSpecFromLpc(gg,xlpc,nmodulations)
                    each_feat[j,:]=mod_spec
                each_feat=np.reshape(each_feat,(1,nfilters*nmodulations))
                feats[i,:]=each_feat
        
            all_feats[seg_id]=feats
    dict2Ark(all_feats,outfile,kaldi_cmd)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract FDLP Modulation Spectral Features with segment files')
    parser.add_argument('scp', help='"scp" list')
    parser.add_argument('segment', help='segment file')
    parser.add_argument('outfile', help='output file')
    parser.add_argument('--nfilters', type=int, default=15,help='number of filters (15)')
    parser.add_argument('--nmodulations', type=int, default=12,help='number of modulations of the modulation spectrum (12)')
    parser.add_argument('--order', type=int, default=50,help='LPC filter order (50)')
    parser.add_argument('--fduration', type=float, default=0.5,help='Window length (0.5 sec)')
    parser.add_argument('--frate', type=int, default=100,help='Frame rate (100 Hz)')
    parser.add_argument('--add_reverb', help='input "clean" OR "small_room" OR "large_room"')
    parser.add_argument('--set_unity_gain', action='store_true', help='Set LPC gain to 1 (True)')
    parser.add_argument('--kaldi_cmd', help='Kaldi command to use to get ark files')
    args = parser.parse_args()
    
    start_time=time.time()
    
    print('%s: Extracting features....' % sys.argv[0])
    sys.stdout.flush()
    getFeats(args)
    
    time_note='Execution Time: {t:.3f} seconds'.format(t=time.time()-start_time)  
    print(time_note)
    sys.stdout.flush()
