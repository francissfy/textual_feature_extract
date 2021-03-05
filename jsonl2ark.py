import argparse
import json
import numpy as np
from kaldiio import WriteHelper
from typing import Tuple


def json2np(line: str) -> Tuple[int, np.ndarray]:
    line_json = json.loads(line)
    lid = line_json["linex_index"]
    feats = line_json["features"]
    # fixme: single layer
    feats = [x["layers"][0]["values"] for x in feats]
    return lid, np.array(feats, dtype=np.float32)


def main(args: argparse.Namespace):
    idx2id = {}
    with open(args.lid, "r") as f:
        ids = f.readlines()
        ids = [lid.strip() for lid in ids]
        ids = [lid for lid in ids if lid != ""]
        for i, lid in enumerate(ids):
            idx2id[i] = lid
    with open(args.jsonl, "r") as f, WriteHelper(f"ark,scp: {args.out_name}.ark,{args.out_name}.scp") as writer:
        while True:
            line = f.readline().strip()
            if line == "":
                break
            lid, feats_np = json2np(line)
            idname = idx2id[lid]
            writer(idname, feats_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str)
    parser.add_argument("--lid", type=str)
    parser.add_argument("--out_name", type=str)
    args = parser.parse_args()
    main(args)
