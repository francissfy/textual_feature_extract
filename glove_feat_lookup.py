import os
import shutil
import argparse
import logging
import numpy as np
from kaldiio import WriteHelper
from typing import Dict, List


lktable: Dict[str, np.ndarray] = {}
key_mapping: Dict[str, str] = {}
oov_keys = set()
feat_dim = 300


def load_table(table_file: str):
    global lktable
    global feat_dim
    with open(table_file, "r") as gf:
        while True:
            gl_line = gf.readline().strip()
            if gl_line == "":
                break
            tmp = gl_line.split(" ")
            w = tmp[0]
            feat = np.array([float(f) for f in tmp[1:]],dtype=np.float)
            lktable[w] = feat
    feat_dim = next(iter(lktable.values())).shape[0]
    logging.info(f"glove name: {args.glove_file}, feat_dim: {feat_dim}, number of vocab: {len(lktable)}")


def convert_text(utt2phone: str, dst_dir: str):
    global lktable
    global key_mapping
    global oov_keys
    global feat_dim
    with open(utt2phone, "r") as uttf: # WriteHelper(f"ark,scp:feats.ark,feats.scp") as ark_writer:
        logging.info(f"converting file: {utt2phone}")
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
                    elif word in oov_keys:
                        similar_key = None
                    else:
                        # looking for similar key
                        if word.endswith("'s"):
                            # remove 's and look for solution
                            striped_word = word[:-2]
                            if striped_word in lktable:
                                similar_key = striped_word
                                # cache key
                                key_mapping[word] = striped_word
                            else:
                                # use blurred search
                                for k in lktable.keys():
                                    if abs(len(k) - len(word)) < len(word) / 2 and (
                                            k.startswith(word) or word.startswith(k)):
                                        key_mapping[word] = k
                                        similar_key = k
                                        break
                    if similar_key is not None:
                        feat = lktable[similar_key]
                    else:
                        oov_keys.add(word)
                        feat = np.zeros(feat_dim, dtype=np.float)
                word_feats.append(feat)
            assert len(word_feats) == len(word_list)
            ark_writer(lid, np.array(word_feats, dtype=np.float))
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        if os.path.isfile("feats.scp"):
            shutil.move("feats.scp", dst_dir)
        if os.path.isfile("feats.ark"):
            shutil.move("feats.ark", dst_dir)
        logging.info(f"done processing {utt2phone}")


def main(args: argparse.Namespace):
    global lktable
    global key_mapping
    global oov_keys
    load_table(args.glove_file)
    for text_dir in args.text_dirs:
        utt2phone_file = os.path.join(args.utt_dir, text_dir, "utt2num_phones")
        dst_dir = os.path.join(args.dst_dir, text_dir)
        convert_text(utt2phone_file, dst_dir)
    # logging the similar and oov keys
    for k in key_mapping.keys():
        logging.info(f"similar key: {k} -> {key_mapping[k]}")
    for k in oov_keys:
        logging.info(f"oov key: {k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove-file", type=str, default="./glove.6B/glove.6B.50d.txt")
    parser.add_argument("--utt-dir", type=str)
    parser.add_argument("--text-dirs", type=str, nargs="+")
    parser.add_argument("--dst-dir", type=str, default="./data/tmp")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)