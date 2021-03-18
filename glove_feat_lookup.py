import os
import shutil
import argparse
import logging
import numpy as np
from kaldiio import WriteHelper
from typing import Dict, List


def main(args: argparse.Namespace):
    lktable: Dict[str, np.ndarray] = {}
    with open(args.glove_file, "r") as gf:
        while True:
            gl_line = gf.readline().strip()
            if gl_line == "":
                break
            tmp = gl_line.split(" ")
            w = tmp[0]
            feat = np.array([float(f) for f in tmp[1:]],dtype=np.float)
            lktable[w] = feat
    feat_dim = next(iter(lktable.values())).shape[0]
    logging.info(f"glove name: {args.glove_file}, feat_dim: {feat_dim}")

    key_mapping: Dict[str, str] = {}
    with open(args.utt2num_phones, "r") as uttf, WriteHelper(f"ark,scp:feats.ark,feats.scp") as ark_writer:
        while True:
            utt_line = uttf.readline().strip()
            if utt_line == "":
                break
            lid, wl = utt_line.split(" ", maxsplit=1)
            word_list: List[str] = [t.split("|", maxsplit=1)[0] for t in wl.split(" ")]
            word_feats: List[np.ndarray] = []
            for word in word_list:
                # strip 's or it will report key not found
                word = word.lower()
                if word == "<sil>":
                    # init zero feat
                    feat = np.zeros(feat_dim, dtype=np.float)
                elif word in lktable:
                    feat = lktable[word]
                else:
                    similar_key = None
                    if word in key_mapping:
                        # find key in cache
                        similar_key = key_mapping[word]
                    else:
                        # looking for similar key
                        if word.endswith("'s"):
                            # remove 's and look for solution
                            striped_word = word[:-2]
                            if striped_word in lktable:
                                similar_key = striped_word
                                # cache key
                                key_mapping[word] = striped_word
                                continue
                            else:
                                # use blurred search
                                for k in lktable.keys():
                                    if abs(len(k) - len(word)) < len(word) / 2 and (
                                            k.startswith(word) or word.startswith(k)):
                                        key_mapping[word] = k
                                        similar_key = k
                                        break
                    if similar_key is not None:
                        logging.info(f"find similar key: {word} -> {similar_key}")
                        feat = lktable[similar_key]
                    else:
                        logging.info(f"word: {word} not found in glove file, use ones")
                        feat = np.ones(feat_dim, dtype=np.float)
                        # raise KeyError(f"word: {word.lower()} not found in glove file")
                word_feats.append(feat)
            ark_writer(lid, np.array(word_feats, dtype=np.float))
    if not os.path.exists(args.dst_dir):
        os.mkdir(args.dst_dir)
    if os.path.isfile("feats.ark"):
        shutil.move("feats.ark", args.dst_dir)
    if os.path.isfile("feats.scp"):
        shutil.move("feats.scp", args.dst_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove-file", type=str, default="./glove.6B/glove.6B.50d.txt")
    parser.add_argument("--utt2num_phones", type=str, default="./data/dev/utt2num_phones")
    parser.add_argument("--dst-dir", type=str, default="./data/tmp")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)