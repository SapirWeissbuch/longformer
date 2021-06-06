"""
Usage: <file_name> --inp_fn=INP_FN --out_fn=OUT_FN --max_num_of_paraphrases=MAX_NUM_OF_PARAGRAPHS --seed=SEED
"""
from docopt import docopt
from pathlib import Path
import json
import random
import itertools
from tqdm import tqdm

def partition(lst, n):
    """
    Slices a list into n nearly-equal-length-partitions
    """
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]

def get_paraphrase_dict(data, mode, num_of_paraphrases):
    if mode == "duplicate":
        return {sample["paragraphs"][0]["qas"][0]["qid"]:
                [sample["paragraphs"][0]["qas"][0]["question"]] * num_of_paraphrases
                for sample in data}


if __name__ == "__main__":
    args = docopt(__doc__)
    inp_fn = Path(args["--inp_fn"])
    out_fn = Path(args["--out_fn"])
    mode = "duplicate" # other models should be supported
    max_num_of_paraphrases = int(args["--max_num_of_paraphrases"])
    seed = int(args["--seed"])

    random.seed(seed)

    with open(inp_fn) as f:
        data = json.load(f)["data"]

    out_data = data.copy()
    random.shuffle(out_data)
    sliced_out_data = partition(out_data, max_num_of_paraphrases+1)
    current_num_of_paraphrases = 0
    for item in sliced_out_data:
        paraphrase_dict = get_paraphrase_dict(item, mode, current_num_of_paraphrases)
        for sample in tqdm(item):
            qid = sample["paragraphs"][0]["qas"][0]["qid"]
            sample["paragraphs"][0]["qas"][0]["question_paraphrases"] = paraphrase_dict[qid]
        current_num_of_paraphrases += 1

    out_data = list(itertools.chain.from_iterable(sliced_out_data))

    with open(out_fn, "w+") as f:
        json.dump({"data": out_data,
                   "version": "duplicates"
        },f)
