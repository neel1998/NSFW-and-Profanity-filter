#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import time
import pickle
import torch
import spacy
nlp = spacy.load('en_core_web_sm')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchtext import data
from torchtext import datasets
import torch.nn as nn


# In[30]:





# In[32]:


class RNN(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
				 bidirectional, dropout, pad_idx):
		
		super().__init__()
		
		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
		
		self.rnn = nn.LSTM(embedding_dim, 
						   hidden_dim, 
						   num_layers=n_layers, 
						   bidirectional=bidirectional, 
						   dropout=dropout)
		
		self.fc = nn.Linear(hidden_dim * 2, output_dim)
		
		self.dropout = nn.Dropout(dropout)
		
	def forward(self, text, text_lengths):
		
		#text = [sent len, batch size]
		
		embedded = self.dropout(self.embedding(text))
		
		#embedded = [sent len, batch size, emb dim]
		
		#pack sequence
		# lengths need to be on CPU!
		packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)
		
		packed_output, (hidden, cell) = self.rnn(packed_embedded)
		
		#unpack sequence
		output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

		#output = [sent len, batch size, hid dim * num directions]
		#output over padding tokens are zero tensors
		
		#hidden = [num layers * num directions, batch size, hid dim]
		#cell = [num layers * num directions, batch size, hid dim]
		
		#concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
		#and apply dropout
		
		hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
				
		#hidden = [batch size, hid dim * num directions]
			
		return self.fc(hidden)



class ProfanityChecker():
	def __init__(self):
		self.TEXT = data.Field(tokenize = 'spacy',
				  tokenizer_language = 'en_core_web_sm',
				  include_lengths = True)
		LABEL = data.Field()
		fields = {'text': ('text', self.TEXT), 'label': ('label', LABEL)}
		with open('./vocab.pkl', 'rb') as f:
			self.TEXT.vocab = pickle.load(f)
		self.INPUT_DIM = len(self.TEXT.vocab)
		self.EMBEDDING_DIM = 100
		self.HIDDEN_DIM = 256
		self.OUTPUT_DIM = 1
		self.N_LAYERS = 2
		self.BIDIRECTIONAL = True
		self.DROPOUT = 0.5
		self.PAD_IDX = self.TEXT.vocab.stoi[self.TEXT.pad_token]

		self.model = RNN(self.INPUT_DIM, 
					self.EMBEDDING_DIM, 
					self.HIDDEN_DIM, 
					self.OUTPUT_DIM, 
					self.N_LAYERS, 
					self.BIDIRECTIONAL, 
					self.DROPOUT, 
					self.PAD_IDX)
	def init(self):
		pretrained_embeddings = self.TEXT.vocab.vectors
		self.model.embedding.weight.data.copy_(pretrained_embeddings)
		UNK_IDX = self.TEXT.vocab.stoi[self.TEXT.unk_token]

		self.model.embedding.weight.data[UNK_IDX] = torch.zeros(self.EMBEDDING_DIM)
		self.model.embedding.weight.data[self.PAD_IDX] = torch.zeros(self.EMBEDDING_DIM)
		self.model.load_state_dict(torch.load('profane-model.pt', map_location=torch.device('cpu')))

	def predict_sentiment(self, sentence):
		self.model.eval()
		tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
		indexed = [self.TEXT.vocab.stoi[t] for t in tokenized]
		length = [len(indexed)]
		tensor = torch.LongTensor(indexed)
		tensor = tensor.unsqueeze(1)
		length_tensor = torch.LongTensor(length)
		prediction = torch.sigmoid(self.model(tensor, length_tensor))
		return prediction.item()

# m = ProfanityChecker()
# m.init()
# res = m.predict_sentiment("what the fuck")
# print(res)
