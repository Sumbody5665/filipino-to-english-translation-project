from __future__ import print_function
from annoy import AnnoyIndex
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
import io


#just a function to show how much of a loop is done already
def show_loop_progress(counter,length):
	if counter < length - 1 :
		print("%.2f%%"%(100*counter/length) ,end='\r')
	else:
		print("%.2f%%"%(100*counter/length))
	return

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
english_loaded=0
english_list = []
english_vector = AnnoyIndex(word_dim)
for line in fin:
	tokens = line.rstrip().split(' ')
	english[tokens[0]] = np.array(list(map(float, tokens[1:])))
	english_list += [tokens[0]]
	english_vector.add_item(english_loaded,list(map(float, tokens[1:])))
	if words_to_load > 0 :
		show_loop_progress(english_loaded,words_to_load)
	else:
		show_loop_progress(english_loaded,n)
	english_loaded += 1
	if (english_loaded >= words_to_load) and (words_to_load > 0):
		break
english[sos] = np.zeros(word_dim)
english_list += [sos]
english_vector.add_item(english_loaded,np.zeros(word_dim))
english_loaded+=1
english[eos] = 0.5*np.ones(word_dim)
english_list += [eos]
english_vector.add_item(english_loaded,0.5*np.ones(word_dim))
english_loaded+=1
english[oov] = 0.2*np.ones(word_dim)
english_list += [oov]
english_vector.add_item(english_loaded,0.2*np.ones(word_dim))
#oov, sos and eos were set to arbitrary vectors




#building translator model
batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
context_dim = 512 # Latent dimensionality of the encoding space.

encoder_input_layer = Input( shape = (None,word_dim) )
encoder_first_layer = LSTM(context_dim,return_sequences=True)(encoder_input_layer)
encoder_middle_layer = LSTM(context_dim,return_sequences=True)(encoder_first_layer)
__, h_state, c_state = LSTM(context_dim,return_state=True)(encoder_middle_layer)
#discard outputs and keep states
encoder_final_state = [ h_state , c_state ]

decoder_input_layer = Input( shape = (None,word_dim) )
decoder_first_layer = LSTM(context_dim,return_sequences=True,return_state=True)
decoder_outputs,__,__ = decoder_first_layer(decoder_input_layer,initial_state=encoder_final_state)
decoder_dense = Dense(word_dim,activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

auto_decoder_input_layer = Input( shape = (None,word_dim) )
auto_decoder_first_layer = LSTM(context_dim,return_sequences=True,return_state=True)
auto_decoder_outputs,__,__ = auto_decoder_first_layer(auto_decoder_input_layer,initial_state=encoder_final_state)
auto_decoder_dense = Dense(word_dim,activation="softmax")
auto_decoder_outputs = auto_decoder_dense(auto_decoder_outputs)
training_model = Model([encoder_input_layer,decoder_input_layer,auto_decoder_input_layer],[decoder_outputs,auto_decoder_outputs])
training_model.load_weights('checkpoint.h5')

encoder_model = Model(encoder_input_layer,encoder_final_state)
encoder_model.summary()

decoder_input_h_state = Input( shape = (context_dim,))
decoder_input_c_state = Input( shape = (context_dim,))
decoder_initial_states = [ decoder_input_h_state , decoder_input_c_state ]
decoder_outputs, state_h, state_c = decoder_first_layer( decoder_input_layer , initial_state=decoder_initial_states )
decoder_current_states = [ state_h , state_c ]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model( [decoder_input_layer]+decoder_initial_states , [decoder_outputs]+decoder_current_states )
decoder_model.summary()
while True:
	input_sentence = sos+" "+input("Enter filipino sentence to be translated to english:").strip()+" "+eos
	input_sentence = input_sentence.replace(",", " , ")
	input_sentence = input_sentence.replace(".", " . ")
	input_sentence = input_sentence.replace("!", " ! ")
	input_sentence = input_sentence.replace("?", " ? ")
	input_sentence = input_sentence.replace("\"", " \" ")
	input_sentence = input_sentence.lower()

	encoder_input = np.array(list(map(lambda x: filipino.get(x,filipino[oov]),input_sentence.split())))
	flip = True
	if flip:
		encoder_input = flip(encoder_input,1)

	understanding = encoder_model.predict(encoder_input)
	output_vectors = np.zeros((1,1,word_dim))
	output_vectors[0][0] = english[sos]
	decoded_sentence = ""
	max_length = 20
	while True:
		output_tokens, h, c = decoder_model.predict([output_vectors] + understanding)
		output_word = english_list[english_vector.get_nns_by_vector(output_tokens,1)]
		decoded_sentence += output_word + " "
		if len(decoded_sentence.split()) >= max_length or output_word == eos:
			break
		output_vectors = output_tokens
		understanding = [h,c]
