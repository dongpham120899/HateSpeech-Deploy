import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from EnviBERT_model.dataset import process_data
from EnviBERT_model.tokenizer import XLMRobertaTokenizer

# load model enviBERT
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
enviBERT = 'EnviBERT_model/enviBERT'
tokenizer_enviBERT = XLMRobertaTokenizer(enviBERT)
model_enviBERT = torch.load('EnviBERT_model/enviBert_model.pt', map_location='cpu')
model_enviBERT.eval()

def predict_enviBERT(sentence, tokenizer=tokenizer_enviBERT, max_len=256, model=model_enviBERT, device=device):
  data = process_data(sentence, 0, tokenizer, max_len)

  ids = torch.tensor(data["ids"], dtype=torch.long, device=device)
  mask = torch.tensor(data["mask"], dtype=torch.long, device=device)
  token_type_ids = torch.tensor(data["token_type_ids"], dtype=torch.long, device=device)
  ids = torch.unsqueeze(ids, 0)
  mask = torch.unsqueeze(mask, 0)
  token_type_ids = torch.unsqueeze(token_type_ids, 0)

  out = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
  preds = torch.sigmoid(out)

  return preds.squeeze().cpu().detach().numpy()