import argparse
import json
import numpy as np
from kaldiio import WriteHelper
from typing import Tuple, Dict


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
    with open(args.text, "r") as textf:
        text_lines = textf.readlines()
        text_lines = [l.strip() for l in text_lines]
        text_lines = [l for l in text_lines if l != ""]
    with open(args.jsonl, "r") as jsonlf:
        json_lines = jsonlf.readlines()
        json_lines = [l.strip() for l in json_lines]
        json_lines = [l for l in json_lines if l != ""]
    assert len(text_lines) == len(json_lines)
    ret_shapes = []
    with WriteHelper(f"ark,scp:{args.out_name}.ark,{args.out_name}.scp") as writer:
        for text_line, json_line in zip(text_lines, json_lines):
            text_parts = text_line.split(" ")
            lid = text_parts[0]
            words = [w.lower() for w in text_parts[1:]]
            feats_data = json.loads(json_line)["features"]
            offset = 0
            sentence_feats = []
            for word in words:
                while offset < len(feats_data):
                    if word.startswith(feats_data[offset]["token"]):
                        sentence_feats.append(feats_data[offset]["layers"][0]["values"])
                        offset += 1
                        break
                    offset += 1
            assert len(sentence_feats) == len(words)
            np_feats = np.array(sentence_feats, dtype=np.float)
            ret_shapes.append(np_feats.shape)
            writer(lid, np_feats)
    # debug
    # with open("shape.json", "w") as wf:
    #     wf.write(json.dumps(ret_shapes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str)
    parser.add_argument("--text", type=str)
    parser.add_argument("--out-name", type=str)
    args = parser.parse_args()
    main(args)
