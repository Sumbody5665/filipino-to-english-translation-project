from __future__ import print_function
#from keras.models import Model
#from keras.layers import Input, LSTM, Dense
import numpy as np
import io

#just a function to show how much of a loop is done already
def show_loop_progress(counter,length):
	if counter < length - 1 :
		print("%.2f%%"%(100*counter/length) ,end='\r')
	else:
		print("%.2f%%"%(100*counter/length))
	return

#path to data file
filename = "sentences.txt"
#start and end of sentence tokens
sos = "<SOS>"
eos = "<EOS>"
oov = "<OOV>"
input = []
output = []

print("Transferring sentences from the file into a list of lists of words...")

with open(filename,encoding="utf8") as f:
	sentence_pairs = f.readlines()
#removes the start of file (not sure yet if that's what it is) character
sentence_pairs[0] = sentence_pairs[0].strip("\ufeff")

sentences_to_load = -1
#set to how many sentences you want to load
#set to negative values if you want to load all sentences
if sentences_to_load>0:
	print("Sentences to load: %s"%(sentences_to_load))
else: 
	print("Sentences to load: all of them")
for x in range(len(sentence_pairs)):
	pair = sentence_pairs[x]
	pair = pair.replace(",", " , ")
	pair = pair.replace(".", " . ")
	pair = pair.replace("!", " ! ")
	pair = pair.replace("?", " ? ")
	pair = pair.replace("\"", " \" ")
	pair = pair.lower()
	left , right = pair.split("\t")
	output = output + [[sos]+left.strip().split()+[eos]]
	input = input + [[sos]+right.strip().split()+[eos]]
	if sentences_to_load > 0 :
		show_loop_progress(x,sentences_to_load)
	else:
		show_loop_progress(x,len(sentence_pairs))
	x = x + 1
	if (x >= sentences_to_load) and (sentences_to_load > 0):
		break
print("%s Filipino sentences and %s English sentences loaded"%(len(input),len(output)))


print("Looking for maximum input sentence length...")
input_max_word_count = 0
for x in range(len(input)):
	if len(input[x]) > input_max_word_count:
		input_max_word_count = len(input[x])
	show_loop_progress(x,len(input))
print("Looking for maximum output sentence length...")
output_max_word_count = 0
for x in range(len(output)):
	if len(output[x]) > output_max_word_count:
		output_max_word_count = len(output[x])
	show_loop_progress(x,len(output))
print("Maximum input sentence length is %s words"%input_max_word_count)
print("Maximum output sentence length is %s words"%output_max_word_count)


words_to_load = -1
#set to how many words you want to load
#set to negative values if you want to load all words
if words_to_load>0:
	print("Words to load: %s"%(words_to_load))
else: 
	print("Words to load: all of them")
print("Loading filipino word vectors...")
fin = io.open('cc.tl.300.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
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
fin = io.open('wiki-news-300d-1M-subword.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
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

#initialization of input and output tensors
encoder_input = np.zeros((len(input),input_max_word_count,word_dim),dtype="float32")
target_decoder_input = np.zeros((len(output),output_max_word_count,word_dim),dtype="float32")
target_decoder_output = np.zeros((len(output),output_max_word_count,word_dim),dtype="float32")
#our model will also include an extra, auto-encoder output for regularization of the encoder network
auto_decoder_input = np.zeros((len(input),input_max_word_count,word_dim),dtype="float32")
auto_decoder_output = np.zeros((len(input),input_max_word_count,word_dim),dtype="float32")

print("Encoding filipino sentences into vectors...")
for x in range(len(input)):
	for y in range(len(input[x])):
		encoder_input[x][y] = filipino.get(input[x][y],filipino[oov])
		auto_decoder_input[x][y] = filipino.get(input[x][y],filipino[oov])
		if y < len(input[x])-1:
			auto_decoder_output[x][y] = filipino.get(input[x][y+1],filipino[oov])
	show_loop_progress(x,len(input))

print("Encoding english sentences into vectors...")
for x in range(len(output)):
	for y in range(len(output[x])):
		target_decoder_input[x][y] = english.get(output[x][y],english[oov])
		if y < len(output[x])-1:
			target_decoder_output[x][y] = english.get(output[x][y+1],english[oov])
	show_loop_progress(x,len(output))

print("Flipping encoder input...")
encoder_input = np.flip(encoder_input,1)
print("Encoder input has been flipped")

print("encoder input shape: %s"% (encoder_input.shape,))
print("target decoder input shape: %s"% (target_decoder_input.shape,))
print("target decoder output shape: %s"% (target_decoder_output.shape,))
print("auto decoder input shape: %s"% (auto_decoder_input.shape,))
print("auto decoder output shape: %s"% (auto_decoder_output.shape,))

