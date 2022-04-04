import os
import re
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

data_path = "cleaned_data_for_ctm.csv"

data = pd.read_csv(data_path).dropna()

def create_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def predict(self, texts, max_input_length=2005, max_output_length=100):
        if type(texts) == list:
            texts = [self.WHITESPACE_HANDLER(text) for text in texts]
        else:
            texts = [self.WHITESPACE_HANDLER(texts)]

        input_ids = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_input_length
        )["input_ids"]

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=max_output_length,
            no_repeat_ngram_size=2,
            num_beams=4
        )

        summaries = []
        for output_id in output_ids:
            summaries.append(self.tokenizer.decode(
                output_id,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ))

        return summaries

model_name = "csebuetnlp/mT5_multilingual_XLSum"
model = Model(model_name)

saving_path = './summarized_texts'

if not os.path.exists(saving_path):
    os.mkdir(saving_path)

chunks = list(create_chunks(data['text'].values.tolist(), 16))

for idx, texts in enumerate(tqdm(chunks)):
    if os.path.exists(f"{saving_path}/chunk{idx}.csv"):
        continue
    
    predictions = model.predict(texts)
    pd.DataFrame(predictions, columns=['summarized_text']).to_csv(f'{saving_path}/chunk{idx}.csv', index=False)
