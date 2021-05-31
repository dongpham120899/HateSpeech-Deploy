import pandas as pd
import numpy as np
import torch
from underthesea import word_tokenize
from transformers import RobertaTokenizer, BertConfig, AutoTokenizer, RobertaConfig, AutoConfig


# load model phoBERT
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
phoBERT = 'vinai/phobert-base'
tokenizer_phoBERT = AutoTokenizer.from_pretrained(phoBERT, use_fast=False)
model_phoBERT = torch.load('PhoBERT_model/phoBert.pt', map_location='cpu')
model_phoBERT.eval()

def convert_samples_to_ids(texts, tokenizer, max_seq_length, labels=None):
    input_ids, attention_masks = [], []

    for text in texts:
        inputs = tokenizer.encode_plus(text, padding='max_length', max_length=max_seq_length, truncation=True)
        input_ids.append(inputs['input_ids'])
        masks = inputs['attention_mask']
        attention_masks.append(masks)

    if labels is not None:
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long), torch.tensor(
            labels, dtype=torch.long)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long)


def predict_phoBERT(sentences, model=model_phoBERT, tokenizer=tokenizer_phoBERT, max_seq_len=256, device=device):
    # sentences = word_tokenize(sentences, format="text")
    input_ids, attention_masks = convert_samples_to_ids([sentences], tokenizer, max_seq_len)
    model.eval()

    y_pred = model(input_ids.to(device), attention_masks.to(device), token_type_ids=None )
    y_pred = y_pred.squeeze().detach().cpu().numpy()

    return y_pred