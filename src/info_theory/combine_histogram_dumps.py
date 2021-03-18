#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""

@author: samiksadhu
"""

'Computing Joint Histogram of features and Labels'

import argparse
import os
import pickle as pkl

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Combine histogram dumps')
    parser.add_argument('out_dir', help='Dump directory')
    args = parser.parse_args()

    pkl_files = [os.path.join(args.out_dir, f) for f in os.listdir(args.out_dir) if f.endswith('.pkl')]

    dist = pkl.load(open(pkl_files[0], 'rb'))
    for pfile in pkl_files[1:]:
        dist += pkl.load(open(pfile, 'rb'))

    dist += 0.0000000000001
    pkl.dump(dist, open(args.out_dir+'/combined.all', 'wb'))
