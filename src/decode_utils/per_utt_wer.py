import argparse
import subprocess
import os
import pickle as pkl


def get_args():
    parser = argparse.ArgumentParser("Compute per utterance WER form Kaldi decode folder")
    parser.add_argument("decode_folder", help="Kaldi decode directory")
    parser.add_argument("save_wer", help="Path to save the per utterance WER")

    return parser.parse_args()


def run(config):
    shell_cmd = 'cat ' + os.path.join(config.decode_folder, 'scoring_kaldi/wer_details/per_utt') + ' | grep csid'
    r = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE)
    r = r.stdout.decode('utf-8').split('\n')

    wer_dict = {}
    for utt in r:
        if utt:
            details = utt.split()
            wer_dict[details[0]] = [(float(details[3]) + float(details[4]) + float(details[5])) * 100 / (
                        float(details[2]) + float(details[3]) + float(details[5])), float(details[2]), float(details[3]), float(details[4]), float(details[5])]

    return wer_dict

if __name__ == "__main__":
    config = get_args()
    wer_dict = run(config)

    with open(config.save_wer, 'wb') as f:
        pkl.dump(wer_dict, f)
