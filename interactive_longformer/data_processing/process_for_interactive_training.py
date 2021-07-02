"""
Usage: <file_name> --inp_fn=INP_FN --out_fn=OUT_FN --max_num_of_paraphrases=MAX_NUM_OF_PARAGRAPHS --seed=SEED [--cache_dir=CACHE_DIR] --mode=MODE
"""
from docopt import docopt
from pathlib import Path
import json
import random
import itertools
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

CACHE_DIR = "/cs/labs/gabis/sapirweissbuch/projects/TeacherFeedbackTrainingProject/longformer/other_models/pegasus"
PEGASUS_MODEL_NAME = 'tuner007/pegasus_paraphrase'
EN_2_DE_MODEL_NAME = "facebook/wmt19-en-de"
DE_2_EN_MODEL_NAME = "facebook/wmt19-de-en"
TORCH_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 5

def get_response(input_text,num_return_sequences,num_beams, model, tokenizer):
    batch_size = len(input_text)
    batch = tokenizer(input_text,truncation=True,padding='longest',max_length=60, return_tensors="pt").to(TORCH_DEVICE)
    translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

def back_translate(input_text, num_return_sequences, num_beams, model1_name, model2_name, cache_dir):
    tokenizer1 = AutoTokenizer.from_pretrained(model1_name, cache_dir=cache_dir)
    model1 = AutoModelForSeq2SeqLM.from_pretrained(model1_name, cache_dir=cache_dir).to(TORCH_DEVICE)
    tokenizer2 = AutoTokenizer.from_pretrained(model2_name, cache_dir=cache_dir)
    model2 = AutoModelForSeq2SeqLM.from_pretrained(model2_name, cache_dir=cache_dir).to(TORCH_DEVICE)
    import pdb; pdb.set_trace()
    translated_chunks = list(chunks(get_response(input_text, num_return_sequences, num_beams, model1, tokenizer1), num_return_sequences))
    back_translated = [[] for i in range(BATCH_SIZE)]
    for i, translated_chunk in enumerate(translated_chunks):
        for sentence in translated_chunk:
            back_translated[i].extend(get_response([sentence], 1, num_beams, model2, tokenizer2))
    return back_translated

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def partition(lst, n):
    """
    Slices a list into n nearly-equal-length-partitions
    """
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]



def get_paraphrase_dict(data, mode, num_of_paraphrases, cache_dir):
    if mode == "duplicate":
        return {sample["paragraphs"][0]["qas"][0]["qid"]:
                [sample["paragraphs"][0]["qas"][0]["question"]] * num_of_paraphrases
                for sample in data}
    else:
        if num_of_paraphrases == 0:
            return {sample["paragraphs"][0]["qas"][0]["qid"]:
                [] for sample in data}
        if mode == "pegasus":
            tokenizer = AutoTokenizer.from_pretrained(PEGASUS_MODEL_NAME, cache_dir=cache_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(PEGASUS_MODEL_NAME, cache_dir=cache_dir).to(TORCH_DEVICE)
            output = {}
            for batch in tqdm(list(chunks(data, BATCH_SIZE))):
                original_questions = [sample["paragraphs"][0]["qas"][0]["question"] for sample in batch]
                paraphrases = list(chunks(get_response(original_questions, num_of_paraphrases, num_of_paraphrases, model, tokenizer), num_of_paraphrases))
                for sample, paras in zip(batch, paraphrases):
                    output[sample["paragraphs"][0]["qas"][0]["qid"]] = paras
            return output

        elif mode == "back-translate":
            output = {}
            for batch in tqdm(list(chunks(data, BATCH_SIZE))):
                original_questions = [sample["paragraphs"][0]["qas"][0]["question"] for sample in batch]
                paraphrases = back_translate(original_questions, num_of_paraphrases, num_of_paraphrases, EN_2_DE_MODEL_NAME, DE_2_EN_MODEL_NAME, cache_dir)
            return output


if __name__ == "__main__":
    args = docopt(__doc__)
    inp_fn = Path(args["--inp_fn"])
    out_fn = Path(args["--out_fn"])
    mode = args["--mode"]
    max_num_of_paraphrases = int(args["--max_num_of_paraphrases"])
    cache_dir = args["--cache_dir"] if args["--cache_dir"] else CACHE_DIR

    seed = int(args["--seed"])

    random.seed(seed)

    with open(inp_fn) as f:
        data = json.load(f)["data"]

    out_data = data.copy()[:10]
    random.shuffle(out_data)
    sliced_out_data = partition(out_data, max_num_of_paraphrases+1)
    current_num_of_paraphrases = 0
    for item in sliced_out_data:
        paraphrase_dict = get_paraphrase_dict(item, mode, current_num_of_paraphrases, cache_dir)
        for sample in tqdm(item):
            qid = sample["paragraphs"][0]["qas"][0]["qid"]
            sample["paragraphs"][0]["qas"][0]["question_paraphrases"] = paraphrase_dict[qid]
        current_num_of_paraphrases += 1

    out_data = list(itertools.chain.from_iterable(sliced_out_data))

    with open(out_fn, "w+") as f:
        json.dump({"data": out_data,
                   "version": "duplicates"
        },f)
