import torch
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

cache_dir = "/cs/labs/gabis/sapirweissbuch/projects/TeacherFeedbackTrainingProject/longformer/other_models/pegasus"
pegasus_model_name = 'tuner007/pegasus_paraphrase'
en_to_de_model_name = "facebook/wmt19-en-de"
de_to_en_model_name = "facebook/wmt19-de-en"
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_response(input_text,num_return_sequences,num_beams, model_name, cache_dir):
    batch_size = len(input_text)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir).to(torch_device)
    batch = tokenizer(input_text,truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

sentence = "In what Eastern European country did violence arise against Uzbeks during 2010?"

pegasus_responses = get_response([sentence], 5, 5, pegasus_model_name, cache_dir)

german_responses = get_response([sentence], 5, 5, en_to_de_model_name, cache_dir)

back_translated = []
for response in tqdm(german_responses):
    back_translated.append(get_response([response], 1, 5, de_to_en_model_name, cache_dir))

print(pegasus_responses)
print()
print(back_translated)
