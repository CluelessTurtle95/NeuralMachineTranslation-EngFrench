# -*- coding: utf-8 -*-

from __future__ import division , print_function , unicode_literals 

import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import tensorflow.compat.v1.keras as keras
# import tensorflow.compat.v1.keras.layers as layers
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import re
import numpy as np
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from sklearn.model_selection import train_test_split
import sys
import pickle

print(tf.__version__)

#filename = "/home/parth/Machine Learning/Datasets/fra.txt"
filename = "fra.txt"
#file_checkpoint = "/home/parth/Machine Learning/Model Checkpoints/English To French/Double Layer/#model.ckpt-22.index"
file_checkpoint = "model.ckpt-22.index"
max_examples = -1

def preprocess_sentence(sent):
    sent = sent.lower().strip()
    sent  = re.sub(r'([?.!’,])' , r' \1'  , sent)
    sent = re.sub(r'[" "]+' , " " , sent)
    sent = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ?.!’,]+' , " " , sent )
    sent = sent.lower().strip()
    sent = "<start> " + sent + " <end>"
    return sent

def create_dataset(file):
    data = open(file).read().strip()
    lines = data.split("\n")
    lines = lines[:max_examples]
    data_sep = [line.split("\t") for line in lines]
    en = [var[0] for var in data_sep]
    fr = [var[1] for var in data_sep]
    for ind , phrase in enumerate(en) :
        en[ind] = preprocess_sentence(phrase)
    for ind , phrase in enumerate(fr) :
        fr[ind] = preprocess_sentence(phrase)
    return en  , fr

class Language():
    def __init__(self , lang):
        self.lang = lang
        self.word2indx = {}
        self.indx2word = {}
        self.vocab = set()
        self.create_indx()
    def create_indx(self):
        for phrase in self.lang :
            self.vocab.update(phrase.split(" "))
        self.vocab = sorted(self.vocab)
        self.word2indx["<pad>"] = 0
        for ind , word in enumerate(self.vocab) :
            self.word2indx[word] = ind + 1
        self.indx2word = {ind : word for word , ind in self.word2indx.items()}

def GRU(num_units):
    gru = layers.GRU(num_units , return_sequences=True, 
                            return_state=True, reset_after = True,
                            #recurrent_activation='sigmoid' , 
                            recurrent_initializer='glorot_uniform')
    return gru

class Encoder(keras.Model):
    def __init__(self , vocab_size , embedding_dim , n_neurons , batch_size):
        super(Encoder , self).__init__()
        self.embedding = layers.Embedding(vocab_size , embedding_dim)
        self.gru1 = GRU(n_neurons)
        self.gru2 = GRU(n_neurons // 2)
        self.batch_size = batch_size
        self.n_neurons = n_neurons
    
    def call(self , x , hidden1 , hidden2):
        embeded = self.embedding(x)
        #embeded = tf.expand_dims(embeded , axis=0) testing with one instance
        outputs , state1 = self.gru1(embeded , initial_state=hidden1)
        outputs , state2 = self.gru2(outputs , initial_state=hidden2)
        return outputs , state1 , state2
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.n_neurons))

class Decoder(keras.Model):
    def __init__(self, vocab_size , batch_size , embedding_dim, dec_units ):
        super(Decoder , self).__init__()
        
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.vocab_size = vocab_size
        self.embeding_size = embedding_dim
        
        self.gru1 = GRU(self.dec_units)
        self.gru2 = GRU(self.dec_units // 2)
        
        self.FCf = layers.Dense(self.vocab_size)
        self.embedding = layers.Embedding(self.vocab_size , embedding_dim )
        
        self.V = layers.Dense(1)
        self.W1 = layers.Dense(self.dec_units)
        self.W2 = layers.Dense(self.dec_units)
        self.W3 = layers.Dense(self.dec_units)
        
    def call(self , inputs , encoder_outputs , state1 , state2):
        
        state1 = tf.expand_dims(state1 , 1)
        state2 = tf.expand_dims(state2 , 1)
        score = self.V(tf.nn.tanh(self.W1(encoder_outputs) + self.W2(state2) + self.W3(state1)))
        score = tf.nn.softmax(score , axis=1)
        
        embeded_input = self.embedding(inputs)
        context = tf.reduce_mean( score * encoder_outputs  , axis=1)
        att_vector = tf.concat([tf.expand_dims(context , 1) , embeded_input ] , axis=2)
        
        output , state1_next = self.gru1(att_vector)
        output , state2_next = self.gru2(output)
        output = self.FCf(tf.reshape(output , [-1 , output.shape[2] ] ))
        return output , state1_next , state2_next
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.dec_units))

def loss_function(real , predicted):
    mask = 1 - np.equal(real , 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real , logits=predicted) * mask
    return tf.reduce_mean(loss_)

def prepareSequences(data , languages):
    eng_sequences , fr_sequences = [ [[lang.word2indx[word] for word in phrase.split(" ") ]for phrase in phrases] for phrases , lang in zip(data , languages)]
    eng_prepared  = pad_sequences(eng_sequences , padding="post" , maxlen=30)
    fr_prepared = pad_sequences(fr_sequences , padding="post" , maxlen=30)
    eng_train , eng_val , fr_train , fr_val = train_test_split(eng_prepared , fr_prepared , test_size = 0.02 )
    return eng_train , eng_val , fr_train , fr_val

def prepareDataset(batch_size , eng_train , fr_train ):
    dataset = tf.data.Dataset.from_tensor_slices( ( eng_train, fr_train ) ).shuffle(num_examples)
    dataset = dataset.batch(batch_size , drop_remainder=True)
    return dataset

def training(encoder , decoder , optimizer , languages , n_epoch , log_dir ):
    log_file = log_dir
    #writer = tf.contrib.summary.create_summary_file_writer(log_file)
    #with writer.as_default():
    for epoch in range(n_epoch):
        total_loss = 0
        for batch , (inp , targ) in enumerate(dataset):
            start = time.time()
            loss = 0
            hidden1 = tf.zeros(shape=(batch_size, layer_size))
            hidden2 = tf.zeros(shape=(batch_size, layer_size // 2))
            size = 0.0
            with tf.GradientTape() as tape:
                enc_outputs , enc_state1 , enc_state2 = encoder(inp , hidden1 , hidden2)
                accuracy = 0.0
                hidden1 = enc_state1
                hidden2 = enc_state2
                hidden_input = [languages[1].word2indx["<start>"]] * batch_size
                hidden_input = np.array(hidden_input).reshape(batch_size , 1)
                for t in range(1 , targ.shape[1]) :
                    dec_output , hidden1 , hidden2 = decoder(hidden_input , enc_outputs , hidden1 , hidden2)

                    hidden_input = tf.expand_dims(targ[ : , t] , 1)
                    loss += loss_function(targ[: , t] , dec_output)
                    dec_output = tf.math.argmax(dec_output , axis=1)
                    is_padding = tf.equal(targ[: , t]  , languages[1].word2indx["<pad>"])
                    acc = tf.cast(tf.equal(tf.cast(targ[: , t] , tf.int64 ) ,  dec_output) , tf.float32) * (1 - is_padding.numpy())
                    accuracy += tf.reduce_mean(acc)
                    size += tf.cast(tf.reduce_sum(1 - is_padding.numpy()) , tf.float32)
            batch_loss = (loss / int(targ.shape[1]))
            accuracy = (accuracy * tf.cast(targ.shape[1] , tf.float32) / size )
            total_loss += batch_loss
            variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss , variables)
            optimizer.apply_gradients(zip(gradients , variables))
            #display.clear_output(wait=True)
            #tf.contrib.summary.scalar(tensor=loss , name="Loss")
            #tf.contrib.summary.scalar("Accuracy" , accuracy )
            print('Epoch {} Batch {} Loss {:.4f} Accuracy : {} Time : {} '.format(epoch + 1,
                                                        batch,
                                                        batch_loss.numpy(),
                                                        accuracy , time.time() - start))    
        if (epoch + 1) % 1 == 0:
            checkpoint.save("model.ckpt")
        #display.clear_output(wait=True)
        result_epoch = 'Epoch {} Loss {:.4f} Accuracy : {} '.format(epoch + 1,
                                            total_loss / n_batches , accuracy)

def translate(sentence):
    sentence = preprocess_sentence(sentence)
    sentence = sentence.strip().split(" ")
    sentence = [languages[0].word2indx[word] for word in sentence]
    
    sentence = [sentence]
    sentence = pad_sequences(sentence , padding="post" , maxlen=30 )
    
    hidden1 = tf.zeros(shape=(1, layer_size))
    hidden2 = tf.zeros(shape=(1, layer_size // 2))
    
    enc_outputs , state1 , state2 = encoder(sentence , hidden1 , hidden2)
    
    hidden1 = state1
    hidden2 = state2
    inp = [languages[1].word2indx["<start>"]] * 1 
    inp = np.reshape(inp , [1 , 1] )
    translated = []
    for t in range(1 , 30):
        output , hidden1 , hidden2 = decoder(inp , enc_outputs , hidden1 , hidden2)
        inp = tf.math.argmax(output , axis=1 )
        inp = tf.expand_dims(inp , 1)
        translated.append(inp)
    translated = [ind.numpy()[0][0] for ind in translated]
    translated = [languages[1].indx2word[ind] for ind in translated]
    result = []
    for index  , word in enumerate(translated):
        if word != "<end>":
            result.append(word)
        else:
            break
    result = " ".join(result)
    print(result)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--translate" , action='store_true' , default=False
)
parser.add_argument(
    "--train" , action='store_true' , default=False
)
parser.add_argument(
    "--file" , type=str , default=None
)
parser.add_argument(
    "--fileCheckpoint" , type=str , default=None
)
parser.add_argument(
    "--NoContinue" , action='store_false' , default=True
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.file :
        filename = args.file
    if args.fileCheckpoint :
        file_checkpoint = args.fileCheckpoint
    
    dataReady = False

    if not os.path.exists('languageData.p'):
        ## Find Data
        if not os.path.exists('FraDataset.p'):
            if os.path.exists(filename) :
                data = create_dataset(filename)
            else :
                print("NO DATASET FOUND! Exiting")
                exit()
            pickle.dump(data , open('FraDataset.p', 'wb'))
        else:
            data = pickle.load(open('FraDataset.p', 'rb'))
    
        languages = [Language(lang) for lang in data]
        pickle.dump(languages, open('languageData.p', 'wb'))
        dataReady = True    
    else:
        if args.train :
            if os.path.exists('FraDataset.p'):
                data = pickle.load(open('FraDataset.p', 'rb'))
            elif os.path.exists(filename):
                data = create_dataset(filename)
                pickle.dump(data , open('FraDataset.p', 'wb'))
            else:
                print("NO DATASET FOUND! Exiting")
                exit()
            dataReady = True
        languages = pickle.load(open('languageData.p', 'rb'))

    batch_size = 50
    layer_size = 500

    if args.train and dataReady:
        eng_train , eng_val , fr_train , fr_val = prepareSequences( data , languages)
        num_examples = len([ len(seq) for seq in eng_train])
        n_batches = num_examples // batch_size
        dataset = prepareDataset(batch_size , eng_train , fr_train)
    
    encoder=Encoder(len(languages[0].vocab) + 1 , batch_size=batch_size , embedding_dim=200 , n_neurons=layer_size)
    decoder=Decoder(len(languages[1].vocab) + 1 , batch_size=batch_size , embedding_dim=200 , dec_units=layer_size)
    optimizer = tf.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(encoder=encoder , decoder=decoder , optimizer=optimizer)
    
    if not args.NoContinue :
        checkpoint.restore(file_checkpoint)
    if args.train :
        if not dataReady :
            print("Data Not Ready ! Exitting")
            exit()
        n_epoch = 1
        training(encoder , decoder , optimizer , languages , n_epoch , "TFLogs/")
    if args.translate :
        inp = str()
        print("Type in english! Type exit to leave")
        while(inp.lower() != "exit"):
            inp = input(">>")
            try:
                print(translate(inp))
            except KeyError as k:
                print("Word not Interpretted! Error : {}".format(k))