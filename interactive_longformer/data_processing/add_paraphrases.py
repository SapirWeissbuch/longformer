"""
Usage: <file_name> --inp_fn=INP_FN --out_fn=OUT_FN --num_of_paraphrases=NUM_OF_P
"""
from docopt import docopt
from pathlib import Path
import json
from tqdm import tqdm

def get_paraphrase_dict(data, mode):
    if mode == "duplicate":
        return {sample["paragraphs"][0]["qas"][0]["qid"]:
                [sample["paragraphs"][0]["qas"][0]["question"]] * num_of_paraphrases
                for sample in data}


if __name__ == "__main__":
    args = docopt(__doc__)
    inp_fn = Path(args["--inp_fn"])
    out_fn = Path(args["--out_fn"])
    mode = "duplicate" # other models should be supported
    num_of_paraphrases = int(args["--num_of_paraphrases"])

    with open(inp_fn) as f:
        data = json.load(f)["data"]

    paraphrase_dict = get_paraphrase_dict(data, mode)
    out_data = data.copy()
    for sample in tqdm(out_data):
        qid = sample["paragraphs"][0]["qas"][0]["qid"]
        sample["paragraphs"][0]["qas"][0]["question_paraphrases"] = paraphrase_dict[qid]

    with open(out_fn, "w+") as f:
        json.dump({"data": out_data,
                   "version": "duplicates"
        },f)
