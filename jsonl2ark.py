import os
import shutil
import argparse
import json
import numpy as np
from kaldiio import WriteHelper
from typing import Tuple, List


def json2np(line: str) -> Tuple[int, np.ndarray]:
    line_json = json.loads(line)
    lid = line_json["linex_index"]
    feats = line_json["features"]
    ret_feats = []
    for feat in feats:
        if feat["token"] == "[CLS]" or feat["token"] == "[SEP]":
            continue
        elif feat["token"].startswith("#"):
            continue
        else:
            ret_feats.append(feat["layers"][0]["values"])
    return lid, np.array(ret_feats, dtype=np.float32)


def main(args: argparse.Namespace):
    with open(args.jsonl, "r") as jsonlf:
        json_lines = jsonlf.readlines()
        json_lines = [l.strip() for l in json_lines]
        json_lines = [l for l in json_lines if l != ""]
    with open(args.utt2num_phones, "r") as numpf:
        utt2num_phones_lines = numpf.readlines()
        utt2num_phones_lines = [l.strip() for l in utt2num_phones_lines]
        utt2num_phones_lines = [l for l in utt2num_phones_lines if l != ""]

    assert len(json_lines) == len(utt2num_phones_lines)
    # NOTE bert embed dimension
    bert_embed_dim = 256

    ret_shapes = []
    with WriteHelper(f"ark,scp:feats.ark,feats.scp") as writer:
        for json_line, utt2num_phones_line in zip(json_lines, utt2num_phones_lines):
            word_list: List[str] = []
            lid, num_phones = utt2num_phones_line.split(" ", maxsplit=1)
            num_phones = num_phones.split(" ")
            for pair in num_phones:
                w, _ = pair.split("|", maxsplit=1)
                word_list.append(w)
            feats_data = json.loads(json_line)["features"]
            # take the average of subwords
            subword_sidx = [0 for _ in word_list]
            widx = 0
            fidx = 0
            while widx < len(word_list):
                if word_list[widx] == "<SIL>":
                    subword_sidx[widx] = -1
                else:
                    while not word_list[widx].lower().startswith(feats_data[fidx]["token"]):
                        fidx += 1
                    subword_sidx[widx] = fidx
                    fidx += 1
                widx += 1
            # take the average over the subwords
            word_feats = []
            for i, sidx in enumerate(subword_sidx):
                if sidx == -1:
                    # SIL
                    feat = np.zeros(256, dtype=np.float)
                else:
                    if i == len(subword_sidx)-1:
                        # last word
                        eidx = len(feats_data)-1
                    elif subword_sidx[i+1] == -1:
                        # next word is SIL
                        if i+1 == len(subword_sidx)-1:
                            # next word is SIL and is end of word seq
                            eidx = len(feats_data)-1
                        else:
                            eidx = subword_sidx[i+2]
                    else:
                        # next word is common word
                        eidx = sidx+1
                    feat = np.mean([f["layers"][0]["values"] for f in feats_data[sidx: eidx]], axis=0, dtype=np.float)
                word_feats.append(feat)
            assert len(word_feats) == len(word_list)
            word_feats = np.array(word_feats, dtype=np.float)
            writer(lid, word_feats)
    if not os.path.exists(args.dst_dir):
        os.mkdir(args.dst_dir)
    if os.path.isfile("feats.ark"):
        shutil.move("feats.ark", args.dst_dir)
    if os.path.isfile("feats.scp"):
        shutil.move("feats.scp", args.dst_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str)
    parser.add_argument("--utt2num-phones", type=str)
    parser.add_argument("--dst-dir", type=str)
    args = parser.parse_args()
    main(args)
