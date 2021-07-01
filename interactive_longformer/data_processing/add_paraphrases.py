"""
Usage: <file_name> --inp_fn=INP_FN --out_fn=OUT_FN --num_of_paraphrases=NUM_OF_P --cache_dir=CAHCE_DIR
"""
from docopt import docopt
from pathlib import Path
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

CACHE_DIR = "/cs/labs/gabis/sapirweissbuch/projects/TeacherFeedbackTrainingProject/longformer/other_models/pegasus"
PEGASUS_MODEL_NAME = 'tuner007/pegasus_paraphrase'
EN_2_DE_MODEL_NAME = "facebook/wmt19-en-de"
DE_2_EN_MODEL_NAME = "facebook/wmt19-de-en"

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_paraphrases(input_text,num_return_sequences,num_beams, model_name, cache_dir):
    batch_size = len(input_text)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir).to(torch_device)
    batch = tokenizer(input_text,truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


def get_paraphrase_dict(data, mode, cache_dir):
    if mode == "duplicate":
        return {sample["paragraphs"][0]["qas"][0]["qid"]:
                [sample["paragraphs"][0]["qas"][0]["question"]] * num_of_paraphrases
                for sample in data}
    else:
        if mode == "pagsus":
            output = {}
            for sample in tqdm(data):
                original_question = sample["paragraphs"][0]["qas"][0]["question"]
                paraphrases = get_paraphrases([original_question], num_of_paraphrases, num_of_paraphrases, model_name, cahce_dir)
                output[sample["paragraphs"][0]["qas"][0]["qid"]] = paraphrases

        else if mode == "back-translate":
            output = {}
            for sample in tqdm(data):
                original_question = sample["paragraphs"][0]["qas"][0]["question"]
                    german_responses = get_response([sentence], 5, 5, en_to_de_model_name, cache_dir)
                    back_translated = []
                    for response in tqdm(german_responses):
                        back_translated.append(get_response([response], 1, 5, de_to_en_model_name, cache_dir))




if __name__ == "__main__":
    args = docopt(__doc__)
    inp_fn = Path(args["--inp_fn"])
    out_fn = Path(args["--out_fn"])
    mode = "duplicate" # other models should be supported
    num_of_paraphrases = int(args["--num_of_paraphrases"])
    cache_dir = args["--cache_dir"] if args["--cache_dir"] else CACHE_DIR
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
