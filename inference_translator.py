from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
import io

#start and end of sentence tokens
sos = "<SOS>"
eos = "<EOS>"
oov = "<OOV>"

input_words_file = "cc.tl.300.vec"
output_words_file = "wiki-news-300d-1M-subword.vec"
words_to_load = -1
#set to how many words you want to load
#set to negative values if you want to load all words
if words_to_load>0:
	print("Words to load: %s"%(words_to_load))
else: 
	print("Words to load: all of them")
print("Loading filipino word vectors...")
fin = io.open(input_words_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
n, word_dim = map(int, fin.readline().split())
filipino = {}
x=0
for line in fin:
	tokens = line.rstrip().split(' ')
	filipino[tokens[0]] = np.array(list(map(float, tokens[1:])))
	if words_to_load > 0 :
		show_loop_progress(x,words_to_load)
	else:
		show_loop_progress(x,n)
	x = x + 1
	if (x >= words_to_load) and (words_to_load > 0):
		break
filipino[sos] = np.zeros(word_dim)
filipino[eos] = 0.5*np.ones(word_dim)
filipino[oov] = 0.2*np.ones(word_dim)
#oov, sos and eos were set to arbitrary vectors

print("Loading english word vectors...")
fin = io.open(output_words_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
n, word_dim = map(int, fin.readline().split())
english = {}
x=0
for line in fin:
	tokens = line.rstrip().split(' ')
	english[tokens[0]] = np.array(list(map(float, tokens[1:])))
	if words_to_load > 0 :
		show_loop_progress(x,words_to_load)
	else:
		show_loop_progress(x,n)
	x = x + 1
	if (x >= words_to_load) and (words_to_load > 0):
		break
english[sos] = np.zeros(word_dim)
english[eos] = 0.5*np.ones(word_dim)
english[oov] = 0.2*np.ones(word_dim)
#oov, sos and eos were set to arbitrary vectors


input_sentence = sos+" "+input("Enter filipino sentence to be translated to english:").strip()+" "+eos
	input_sentence = input_sentence.replace(",", " , ")
	input_sentence = input_sentence.replace(".", " . ")
	input_sentence = input_sentence.replace("!", " ! ")
	input_sentence = input_sentence.replace("?", " ? ")
	input_sentence = input_sentence.replace("\"", " \" ")
	input_sentence = input_sentence.lower()

encoder_input = np.array(list(map(lambda x: filipino[x],input_sentence.split())))


