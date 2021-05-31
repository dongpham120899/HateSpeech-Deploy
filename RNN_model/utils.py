from RNN_model.models import *
from RNN_model.tokenizer import ToxicityDataset
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from RNN_model.preprocessing import preprocess
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


device = 'cuda' if torch.cuda.is_available() else 'cpu' 

file = open('RNN_model/full_comments.txt','r')
docs = file.readlines()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)

model = torch.load('RNN_model/spoken_form_LSTM_model.pt', map_location=device)
model.eval()
# model = NeuralNet(np.zeros((100,100)))
# model.load_state_dict("RNN_model/spoken_form_RNN_model.pt")
# model.eval()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_RNN(sentence):
   sentence = preprocess(sentence)
   x_infer = tokenizer.texts_to_sequences([sentence])
   x_infer = pad_sequences(x_infer, maxlen=400)
   test_preds = np.zeros((200, 7))
   infer = ToxicityDataset(x_infer, test_preds)
   infer_loader = DataLoader(infer, batch_size=1, shuffle=False)
   test_preds = np.zeros((len(infer), 7))
   for i, x_batch in enumerate(infer_loader):
      text = x_batch['text'].to("cpu")
      y_pred = sigmoid(model(text).detach().cpu().numpy())
      test_preds[i * 1:(i+1) * 1, :] = y_pred

   return test_preds[0]


