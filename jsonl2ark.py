import os
import shutil
import argparse
import json
import numpy as np
from kaldiio import WriteHelper
from typing import Tuple
from collections import OrderedDict


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
            word2num_phones = OrderedDict()
            lid, num_phones = utt2num_phones_line.split(" ", maxsplit=1)
            num_phones = num_phones.split(" ")
            for pair in num_phones:
                w, dur = pair.split("|", maxsplit=1)
                word2num_phones[w] = dur
            feats_data = json.loads(json_line)["features"]
            offset = 0
            sentence_feats = []
            for word in word2num_phones.keys():
                if word == "<SIL>":
                    sentence_feats.append(np.zeros(bert_embed_dim, dtype=np.float))
                else:
                    while offset < len(feats_data):
                        if word.lower().startswith(feats_data[offset]["token"]):
                            sentence_feats.append(feats_data[offset]["layers"][0]["values"])
                            offset += 1
                            break
                        offset += 1
            assert len(sentence_feats) == len(word2num_phones.keys())
            np_feats = np.array(sentence_feats, dtype=np.float)
            ret_shapes.append(np_feats.shape)
            writer(lid, np_feats)
            # write the duration file
            with open("bert_embed_dur", "w") as durwf:
                durs = " ".join([str(d) for d in word2num_phones.values()])
                durwf.write(f"{lid} {durs}\n")
    if not os.path.exists(args.dst_dir):
        os.mkdir(args.dst_dir)
    if os.path.isfile("feats.ark"):
        shutil.move("feats.ark", args.dst_dir)
    if os.path.isfile("feats.scp"):
        shutil.move("feats.scp", args.dst_dir)
    if os.path.isfile("bert_embed_dur"):
        shutil.move("bert_embed_dur", args.dst_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str)
    parser.add_argument("--utt2num-phones", type=str)
    parser.add_argument("--dst-dir", type=str)
    args = parser.parse_args()
    main(args)
