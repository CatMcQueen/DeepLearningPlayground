#!/usr/bin/python
# -*- coding: utf-8

import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
from itertools import chain
from numpy.random import randint
import re


def train_test_split(source_file, target_file, n_test=3000, max_sentence_len=30):
    """
    Both text files contain sentences in each language and map between
      each other one to one.
    This method reads the files into memory after eliminating all
      nonstandard english characters.
    Then it ensures that no sentence pairs are kept where one or the other exceeeds
      the maximum sentence length.
    A random sample of the data is set aside for training/validation.
    Corpus objects for both languages are created
    -obtains a list of unique words
    -assigns each an index
    -these along with EOS and SOS are the tokens that
     will be used for training.
    Like the Char-RNN lab using the indices generated in each corpus 
     each sentence's words are mapped to indexes.
    returns:
      testing_pairs: A list of the original sentence pairs (a list of tuples 
                       of two strings each [(spanish_sentence1, english_sentence1), ...] 
      training_pairs: similar but for training
      train_index_pairs: a list of tuples containing the indices of the words
                         in each sentence in the training_pairs 
                         [ ([11, 1592, 350, 279, 438], [2, 219, 38]), ... ]
      test_index_pairs: similar but for training_pairs
      
    If you don't like working with a list of tuples just call
      spanish_testing, english_testing = zip(*testing_pairs) # both lists of strings now
      spanish_training, english_training = zip(*training_pairs) # same
      
      # the following are lists of unequal length integer lists representing indices 
      indexed_spanish_training, indexed_english_training = zip(*train_index_pairs) 
      indexed_spanish_testing, indexed_english_testing = zip(*test_index_pairs)
     
     The only difference until now from the Char-RNN lab is in processing data for two languages
       and limiting the data to sentences that aren't above a certain length.
    """
    
    # reads file and removes all non a-z characters
    def read_file(text_file):
        r1 = lambda x:re.sub(r"[.!?]$", r" <EOS>", x.strip().lower())
        r2 = lambda x:re.sub(r"(-|—)", " ", x)
        r3 = lambda x:re.sub(r"\([^\(]+\)", " ", x)
        
        contents = [r1(r2(r3(l))) for l in open(text_file, "r")]
        acceptable = list(range(97,123)) + \
                     [32, 160, 193, 201, 205, 209, 211, 218, 220,\
                      225, 233, 237, 241, 243, 250, 252]

        contents = [filter(lambda x:ord(x) in acceptable, \
                           line).strip().lower() for line in contents]
        contents = [" ".join(re.split(r"\s+", line)) for line in contents] # removes consecutive spaces
        return contents

    source_lines = read_file(source_file)
    target_lines = read_file(target_file)
    
    n_words = lambda s:len(s.split(" "))
   
        
   
        
    keep_pair_if = lambda pair: 6 < min(n_words(pair[0]),n_words(pair[1])) < max_seq_len and \
                                len(re.findall("(http|www|org|edu|com)", pair[1])) == 0
        
    pairs = zip(source_lines, target_lines)
    pairs = filter(keep_pair_if, pairs)

    # could try training on shortest first. 
    #shortest = lambda pair: min(n_words(pair[0]),n_words(pair[1]))
    #pairs = sorted(pairs, key=shortest)
    
    # filter keeps all elements of pairs
    pairs = filter(keep_pair_if, pairs)
    source_lines, target_lines = zip(*pairs)
    print len(source_lines), 'lines in data'

    source_corpus = Corpus(source_lines)
    target_corpus = Corpus(target_lines)

    n_spanish = source_corpus.corpus_size
    n_english = target_corpus.corpus_size
    all_indexed_pairs = zip(source_corpus.training, target_corpus.training)

    np.random.seed(2)
    test_idc = randint(0, len(pairs), n_test)
    train_idc = set(np.arange(len(pairs))) - set(test_idc)

    # list of 2-string tuples consisting of source / reference sentence pairs
    testing_pairs = map(lambda k: pairs[k], test_idc)
    training_pairs = map(lambda k: pairs[k], train_idc)

    # list of tuples consisting of source / reference sentence pairs 
    # but represented by word indexes
    train_index_pairs = map(lambda k:all_indexed_pairs[k], train_idc)
    test_index_pairs = map(lambda k:all_indexed_pairs[k], test_idc)
    
    return source_lines, target_lines, source_corpus, target_corpus, testing_pairs, training_pairs, train_index_pairs, test_index_pairs


class Corpus():
    def __init__(self, input_lines, n_train=5000):
        self.SOS = 0
        self.EOS = 1
        self.idx_word,         self.word_idx = self.parse_words(input_lines)
        self.n_train = n_train
        
        self.parse_words(input_lines)
        self.corpus_size = len(self.idx_word)
        self.lines = [l.strip().lower() for l in input_lines]
        self.training = [self.sentence_to_index(l) for l in self.lines]
        
    def parse_words(self, lines):
        sls = lambda s: s.strip().lower().split(" ")
        words = sorted(set(list(                 chain(*[sls(l) for l in lines]))))
        words = ["<SOS>", "<EOS>"] + filter(lambda word:set(word).intersection(set(map(chr, range(97,123))))!=set(), words)
        n = 3000
        print(len(words), 'words')
        idx_word = dict(list(enumerate(words)))
        word_idx = dict(zip(words, list(range(len(words)))))
        
        return idx_word, word_idx
    
    def sentence_to_index(self, s):
        words = s.split(" ")
        indices = [self.word_idx[word] for word in words]
        return indices
    
    def index_to_sentence(self, indices):
        return " ".join(self.idx_word[idx] for idx in indices)


# recommend to use this class later, same as nn.Linear
class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.W = torch.nn.Parameter(torch.randn(output_size, input_size))
        self.b = torch.nn.Parameter(torch.randn(output_size, 1))
        
        # use rand(n) to get tensors to intitialize your weight matrix and bias tensor 
        # then use Parameter( ) to wrap them as Variables visible to module.parameters()
        # to use variance scaling initialization just divide the variable like a ndarray
        
    def forward(self, input_var):
        # standard linear layer, no nonlinearity
        return torch.matmul(self.W, input_var) + self.b


"""
encoder topology
-standard GRU topology, see slides for a reveiw
-context vector is the hidden state of the last time step and last layer
-review the char-rnn lab

below every GRUCell is its input, to its left is the hidden input
-note that of all the outputs and final hidden states only one is kept

-h_3 -> GRUCell3 -> GruCell3 -> ... -> GRUCell3 -> context vector
-h_2 -> GRUCell2 -> GruCell2 -> ... -> GRUCell2
-h_1 -> GRUCell1 -> GruCell1 -> ... -> GRUCell1
        emb[0]      emb[1]             emb[n-1]

-use zero Variables as the initial hidden states

Pytorch RNN pipelines and this lab
-Never use one hot encodings in pytorch. Several loss functions, nn.Embedding, 
 and other classes are programmed to use indexed tensors whenever possible
-like tensorflow and GRUCell  (batch, input_dim), (batch, hidden_dim) shaped tensor Variables
 in our case batch = 1 and input_dim = hidden_dim most likely
-nn.Embeddding does the same thing as one hot encoding the input and then running
  it through a linear transformation (so no bias or nonlinearity)
-good idea to specify the max_norm of both embeddings when initialized
"""


class Encoder(nn.Module):
    def __init__(self, hidden_size, source_vocab_size, n_layers=2):
        super(Encoder, self).__init__()
	print "hidden size" , hidden_size
        self.input_size = hidden_size
        self.hidden_size = hidden_size
        self.vocab_size = source_vocab_size
        self.n_layers = n_layers
        
        # multiple ways to do this
        cells = [nn.GRUCell(self.hidden_size, self.hidden_size) for _ in range(n_layers)]
        self.cells = nn.ModuleList(cells)        
        print "self", self.input_size, "hidden_size", self.hidden_size, "layers", self.n_layers
	self.GRU = nn.GRU(self.input_size, self.hidden_size, self.n_layers)
        self.LSTM = nn.LSTM(self.input_size, self.hidden_size, self.n_layers)
        
        self.embedding = nn.Embedding(self.vocab_size, self.input_size, scale_grad_by_freq=True)
        
        # so they're accessible to decoder.cuda()
        self.init_hidden_states = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        self.init_cell_states = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
    

    # if you want to udnerstand this better consider writing
    # implementing tf.contrib.rnn.legacy.MultiCellRNN here
    def time_step(self, input_, hidden_states):
        pass
    
    # consider coding this as rnn_decoder as shown in the Char-RNN lab
    def forward(self, source_variable):
        # discard all outputs except the last one.
        n_timesteps = source_variable.data.cpu().size()[0]
        embedded = self.embedding(source_variable).view(-1, 1, self.hidden_size)
        hidden_states = self.init_hidden_states
        cell_states = self.init_cell_states
        
        """ 1st implementation watch shapes
        hidden_states = [torch.zeros(1, self.hidden_size) for _ in range(self.n_layers)]
        for i in range(n_timesteps):
            out = embedded[i] # GRUCell takes (batch, hidden_dim) not (batch, seq_len, hidden_size)
            for j in range(self.n_layers):
                hidden_states[j] = self.cells[j](out, hidden_states[j])
                out = hidden_states[j]
        return out"""


        """ 2nd implementation
        for i in range(n_timesteps):
             out, hidden_states = self.GRU(self.embedding[i:i+1], hidden_states) 
             #out, (hidden_states, cell_states) = self.LSTM(self.embedding[i:i+1], (hidden_states, cell_states)) 
        
        return out """
        
        # 3rd
        return self.GRU(self.embedding(source_variable).view(-1, 1, self.hidden_size), hidden_states)[0][-1]
        #return self.LSTM(self.embedding(source_variable).view(-1, 1, self.hidden_size), \
        #                 (hidden_states, cell_states))[0][-1]
   

class Decoder(nn.Module):
    def __init__(self, hidden_size, target_vocab_size, n_layers=2, max_target_length=30):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.vocab_size = target_vocab_size
        self.n_layers = n_layers
        self.max_length = max_target_length
        
        # three possible implementations
        self.cells = nn.ModuleList([nn.GRUCell(self.input_size, self.hidden_size) for _ in range(n_layers)])
        self.GRU = nn.GRU(self.input_size, self.hidden_size, self.n_layers)
        self.LSTM = nn.LSTM(self.input_size, self.hidden_size, self.n_layers)
        
        # maps to logits
        self.Linear = Linear(self.hidden_size, self.vocab_size)
        # much better to store words in 1000 dimensional space than in 20000 (one hot encoded) dim space
        self.embedding = nn.Embedding(self.vocab_size, hidden_size, scale_grad_by_freq=True)

        self.loss = nn.CrossEntropyLoss() # here so it's accessible using decoder.cuda()

        
    def forward(self, context, reference_sentence=None):
        # if the reference sentence is given then use it for teacher forcing
        use_teacher_forcing = reference_sentence is not None

        embedded_sos = self.embedding(Variable(torch.LongTensor([[0]]))).view(1, 1, self.hidden_size)

        predictions = []                           
        output_logits= []

        if use_teacher_forcing:
            inputs = [embedded_sos.view(1, 1, hidden_size)] +                      list(torch.split(self.embedding(reference_sentence)                                .view(-1, 1, self.hidden_size)[1:], 1, 0))
            input_ = inputs[0]
        else:
            input_ = embedded_sos

        c_i = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        h_i = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        
        for i in range(self.max_length):            
            #output, (h_i, c_i) = self.LSTM(input_, (h_i, c_i))
            output, h_i = self.GRU(input_, h_i)

            logits = self.Linear(output.view(-1, 1))
            prediction = logits.data.cpu().numpy().argmax()

            predictions.append(prediction)
            if not use_teacher_forcing:
                if prediction == target_corpus.EOS:
                    break
                
                input_ = self.embedding(Variable(torch.LongTensor([[prediction]])))
            else:
                if i == reference_sentence.size()[0]-1:
                    break # don't translate for longer
                input_ = inputs[i+1]
                
            output_logits.append(logits)

        return output_logits, predictions


# changed
def get_loss(output_probs, correct_sentence_indices, predicted_sentence_indices, loss_op):
    """ 
    note that output_probs and correct_indices will often have different shapes
    look up documentaiton on NLLLoss, but if you 
    predicted_inde
    inputs:
    output_probs Variable
    target_sentence_indices
    predicted_sentence_indices
    params:
      output_probs: a list of Variable (not FloatTensor) with the predicted sequence length
      correct_indices: a list or tensor of type int with the same length, will need
                       to be converted to a Variable before compared to the output_probs
    
    """
    
    # changed: Convert correct_sentence_indices to a Variable if it isn't already
    # remember embedded tensors (like your logit probabilities) should have one more dimension than
    #    the indexed tensor that they're compared against
    # compute cross entropy, recommend NLLLoss since it works after LogSoftmax
    #  approx need to call log  on your logit softmax probabilities or use 
    #  max score or LogSoftmax closest to zero for predictions
    # should return a Variable representing the loss
    # reommended: consider light dropout before L1, L2 regularization.
    
    # moved loss to decoder so it would be accessible to decoder.cuda()
 
    if type(output_probs) is list:
        output_probs = torch.cat(output_probs, dim=1).transpose(0,1)
        
    min_length = int(min(output_probs.size()[0], correct_sentence_indices.size()[0]))
    loss = loss_op(output_probs[0:min_length], correct_sentence_indices[0:min_length])
    
    arr1 = correct_sentence_indices.data.cpu().numpy()
    arr2 = np.array(predicted_sentence_indices) # should have been list of ints
    accuracy = float(np.sum(arr1[0:min_length] == arr2[0:min_length])) / arr1.shape[0]

    return loss, accuracy 


def print_output(epoch=None, in_out_ref_sentences=None, stats=None, teacher_forced=False):
    """
    params:
      teacher_forced: whether or not teacher forcing was used
    """  
    # not a great practice to use global
    global source_corpus, target_corpus
    
    if stats is not None:
        s = ": perplexity: %.3f: loss: %.6f, accuracy: %.1f" % (2**stats[0], stats[0], stats[1])
        if teacher_forced:
            print("epoch %d %s - using teacher forcing" % (epoch, s))
        else:
            print("epoch %d iteration %s" % (epoch, s))
    
    if in_out_ref_sentences is not None:
        if epoch is not None:
            print ("Outputs during epoch %d %s" % (epoch, "" if not teacher_forced else "using teacher forcing"))

        source_indices, predicted_indices, reference_indices = in_out_ref_sentences
        print "In:       ", source_corpus.index_to_sentence(source_indices)
        print "Out:      ", target_corpus.index_to_sentence(predicted_indices)
        print "Reference:", target_corpus.index_to_sentence(reference_indices)
        
        
def train(encoder, decoder, training_pairs, testing_pairs, train_index_pairs, test_index_pairs,
                source_corpus, target_corpus, teacher_forcing_ratio, 
                epoch_size, learning_rate, decay, batch_size, print_every):
    """
    Again teacher forcing not required, but will reduce training time and is 
      much simpler to implement.
    You may want to lower the teacher forcing ratio as the number 
      of epochs progresses as it starts to learn word-word connections.
    
    If you wish to use a learning rate schedule, you will need to initialize new optimizers
       every epoch. Note that only a few optimizers let you specify learning rate decay.
    See notes below for help in training / debugging
    Don't hesitate to ask for help1
    
    """
    if torch.cuda.is_available():
        arr2var = lambda sent: Variable(torch.LongTensor(sent)).cuda()
    else:
        arr2var = lambda sent: Variable(torch.LongTensor(sent))
 
    n_test = len(testing_pairs)
    training_var_pairs = [(arr2var(_1), arr2var(_2)) for (_1, _2) in train_index_pairs]
    testing_var_pairs =  [(arr2var(_1), arr2var(_2)) for (_1, _2) in test_index_pairs]
    
    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optim = torch.optim.Adam(all_params, lr=learning_rate, weight_decay=decay)     
    
    # if training on increasing length sentences
    # sentence_id = 0
    batch_loss, batch_acc = [], [] # for printing
    test_every = 20
     
    for i in range(n_epochs):
        for j in range(epoch_size):
            # consider whether or not to use teacher forcing on printing iterations
            use_teacher_forcing = np.random.random() < teacher_forcing_ratio
            
            sentence_id = np.random.randint(0, len(training_pairs))
                
            source, reference = training_var_pairs[sentence_id]
            context = encoder(source)
            ref = reference if use_teacher_forcing else None
            
            # list of Variables, list of int
            output_logits, predictions = decoder(context, reference_sentence=ref)
            
            loss, accuracy = get_loss(output_logits, reference, predictions, decoder.loss)
            
            loss.backward()

            # for reporting teacher forcing doesn't count towards batch training statistics
            if not use_teacher_forcing:
                batch_loss.append(loss.data.cpu().numpy().flatten()[0])
                batch_acc.append(accuracy)
            
            # we could have done loss /= batch_size; loss.backward()
            if (j+1) % batch_size == 0:
                # effective batch updates
                for p in all_params:
                    p.data.div_(batch_size)
                    
                optim.step()
                optim.zero_grad()

                batch_loss_, batch_loss = sum(batch_loss) / (len(batch_loss)+1e-13), []
                batch_acc_, batch_acc = sum(batch_acc) / (len(batch_acc)+1e-13), []

                print("\nEnd of batch %d epoch %d" % (j // batch_size, i+1))
                print_output(epoch=i, stats=(batch_loss_, batch_acc_))
            
            if (j+1) % print_every == 0:
                source_idc, reference_idc = train_index_pairs[sentence_id]
                print_output(in_out_ref_sentences = (source_idc, predictions, reference_idc), teacher_forced=use_teacher_forcing)
        
            if (j+1) % test_every == 0:
                test_loss, test_acc = 0, 0
                for ((src, tgt), (src_str, tgt_str)) in zip(testing_var_pairs, test_index_pairs):
                    probs, predictions = decoder(encoder(src))
                    _1, _2 = get_loss(probs, tgt, predictions, decoder.loss)
                    test_loss += _1.data.cpu().numpy().flatten()[0]
                    test_acc += _2
                test_loss /= n_test
                test_acc /= n_test

                print("\n-- Epoch %d Test Results;: perplexity: %.3f: loss: %.6f, accuracy: %.1f\n" % (i+1, 2**test_loss, test_loss, test_acc))

    return encoder, decoder

# can use this to print out your final translation sentences 
def sample(encoder, decoder, sentence_pairs):#, testing_results = None):
    # never use volatile outside of inference
    if torch.cuda.is_available():
        arr2var = lambda x: Variable(torch.LongTensor(x), volatile=True).cuda()
    else:
        arr2var = lambda x: Variable(torch.LongTensor(x), volatile=True)
    
    sentence_var_pairs =  [(arr2var(_1), arr2var(_2)) for (_1, _2) in sentence_pairs]
    prin
    for (source_sentence, reference_sentence) in sentence_var_pairs:
        context = encoder(source_sentence)
        output_probs, predictions = decoder(reference_sentence)
        print_output(in_out_ref_sentences=(source_sentence, predictions, reference_sentence))

# parameters needed for data processing
source_file = "es.txt"
target_file = "en.txt"
n_test = 30
max_seq_len = 30 # maximum sentence length

# pretty much everything you need, see train_test_split doc string
source_lines, target_lines, source_corpus, target_corpus, testing_pairs, training_pairs, train_index_pairs, test_index_pairs = train_test_split(source_file, target_file, n_test, max_seq_len)

# example hyperparameters
epoch_length = 800
batch_size = 20
n_layers = 2  # 2 or more recommended
learning_rate = .01
decay_rate = .98 # decays every batch_size sentences
print_every = 3
n_epochs = 25
hidden_size = 1024
teacher_forcing_ratio = .5

encoder = Encoder(hidden_size, source_corpus.corpus_size, n_layers)
decoder = Decoder(hidden_size, target_corpus.corpus_size, n_layers, max_target_length=max_seq_len)
use_cuda = torch.cuda.is_available()

# all of our submodules and Variables are class members of encoder and decoder
# so this is the only time we need to call this. If you are using a gpu ensure
# you call .cpu() whenever you want to directly access a tensor's value from code
if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

encoder, decoder = train(encoder, decoder, training_pairs, testing_pairs, \
                         train_index_pairs, test_index_pairs, source_corpus, \
                         target_corpus, teacher_forcing_ratio, epoch_length, \
                         learning_rate, decay_rate, batch_size, print_every)


_="""
Start simple, seriously
-consider implementing it without teacher forcing first.
-just get a scalar loss variable as fast as you can.

Recommend writing new code first inside of a jupyter notebook cell
-Just initialize variables of the correct size as mock inputs
-Can see the Variables after each line of code
-this way getting Variable shapes right will be less of a challenge
-Can use tab for code completion and shift tab to see the signature of
   any torch method when inside the parentheses
-About a majority of the methods are the same as numpy or 
 tensorflow so chances are with a few tries you'll find the right method
-This is also a very good way to learn pytorch

If and only if it's not working
-start by tuning hyperparameters
-batch_size, teacher forcing, learning rate are good places to start
-Make sure your teacher forcing implementation is correct

Training philosophy behind vanilla seq2seq nmt systems (see also sutskever, 2014):
-need to learn somewhat one to one word connections first
-shift to learning long term dependencies by reducing teacher forcing 
-consider a schedule for learning rate and/or teacher forcing ratio

Saving and restoring weights might speed up your workflow significantly
-might want to checkpoint once you've learned word-word connections

Regularization is super simple in pytorch.
-Be careful woing dropout on the hidden states between cells
  see https://arxgiv.org/pdf/1512.05287.pdf.
-low to moderate dropout after both embeddings may be helpful
decoder = Decoder(hidden_size, target_corpus.corpus_size, n_layers, max_target_length=max_seq_len)"""
