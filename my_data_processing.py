# -*- coding: utf-8 -*-
import numpy as np
import re
import random
import json
import collections
import numpy as np
import my_parameters as params
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet as wn 
import os
import pickle
import multiprocessing
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger

FIXED_PARAMETERS, config = params.load_parameters()

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": -1
}

PADDING = "<PAD>"
POS_Tagging = [PADDING, 'WP$', 'RBS', 'SYM', 'WRB', 'IN', 'VB', 'POS', 'TO', ':', '-RRB-', '$', 'MD', 'JJ', '#', 'CD', '``', 'JJR', 'NNP', "''", 'LS', 'VBP', 'VBD', 'FW', 'RBR', 'JJS', 'DT', 'VBG', 'RP', 'NNS', 'RB', 'PDT', 'PRP$', '.', 'XX', 'NNPS', 'UH', 'EX', 'NN', 'WDT', 'VBN', 'VBZ', 'CC', ',', '-LRB-', 'PRP', 'WP']
POS_dict = {pos:i for i, pos in enumerate(POS_Tagging)}

base_path = os.getcwd()
nltk_data_path = base_path + "/../TF/nltk_data"
nltk.data.path.append(nltk_data_path)
stemmer = nltk.SnowballStemmer('english')

tt = nltk.tokenize.treebank.TreebankWordTokenizer()

def load_nli_data(path, snli=False, shuffle = True):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    data = []
    with open(path, encoding='utf-8') as f:
        for line in tqdm(f,ascii=True):
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            data.append(loaded_example)
        if shuffle:
            random.seed(1)
            random.shuffle(data)
    return data

def sentences_to_padded_index_sequences(datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    # Extract vocabulary
    def tokenize(string):
        string = re.sub(r'\(|\)', '', string)
        return string.split()

    
    

    word_counter = collections.Counter()
    char_counter = collections.Counter()
    # mgr = multiprocessing.Manager()
    # shared_content = mgr.dict()
    # process_num = config.num_process_prepro
    # process_num = 1
    for i, dataset in enumerate(datasets):
        # if not shared_file_exist:
        #     num_per_share = len(dataset) / process_num + 1
        #     jobs = [ multiprocessing.Process(target=worker, args=(shared_content, dataset[i * num_per_share : (i + 1) * num_per_share] )) for i in range(process_num)]
        #     for j in jobs:
        #         j.start()
        #     for j in jobs:
        #         j.join()

        for example in tqdm(dataset,ascii=True):
            s1_tokenize = tokenize(example['sentence1_binary_parse'])
            s2_tokenize = tokenize(example['sentence2_binary_parse'])

            word_counter.update(s1_tokenize)
            word_counter.update(s2_tokenize)

            for i, word in enumerate(s1_tokenize):
                char_counter.update([c for c in word])
            for word in s2_tokenize:
                char_counter.update([c for c in word])

        # shared_content = {k:v for k, v in shared_content.items()}



    


    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    if config.embedding_replacing_rare_word_with_UNK: 
        vocabulary = [PADDING, "<UNK>"] + vocabulary
    else:
        vocabulary = [PADDING] + vocabulary
    # print(char_counter)
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    indices_to_words = {v: k for k, v in word_indices.items()}
    char_vocab = set([char for char in char_counter])
    char_vocab = list(char_vocab)
    char_vocab = [PADDING] + char_vocab
    char_indices = dict(zip(char_vocab, range(len(char_vocab))))
    indices_to_char = {v: k for k, v in char_indices.items()}
    

    for i, dataset in enumerate(datasets):
        for example in tqdm(dataset,ascii=True):
            for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
                example[sentence + '_index_sequence'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)
                example[sentence + '_inverse_term_frequency'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.float32)

                token_sequence = tokenize(example[sentence])
                padding = FIXED_PARAMETERS["seq_length"] - len(token_sequence)
                      
                for i in range(FIXED_PARAMETERS["seq_length"]):
                    if i >= len(token_sequence):
                        index = word_indices[PADDING]
                        itf = 0
                    else:
                        if config.embedding_replacing_rare_word_with_UNK:
                            index = word_indices[token_sequence[i]] if word_counter[token_sequence[i]] >= config.UNK_threshold else word_indices["<UNK>"]
                        else:
                            index = word_indices[token_sequence[i]]
                        itf = 1 / (word_counter[token_sequence[i]] + 1)
                    example[sentence + '_index_sequence'][i] = index
                    
                    example[sentence + '_inverse_term_frequency'][i] = itf
                
                example[sentence + '_char_index'] = np.zeros((FIXED_PARAMETERS["seq_length"], config.char_in_word_size), dtype=np.int32)
                for i in range(FIXED_PARAMETERS["seq_length"]):
                    if i >= len(token_sequence):
                        continue
                    else:
                        chars = [c for c in token_sequence[i]]
                        for j in range(config.char_in_word_size):
                            if j >= (len(chars)):
                                break
                            else:
                                index = char_indices[chars[j]]
                            example[sentence + '_char_index'][i,j] = index 
    

    return indices_to_words, word_indices, char_indices, indices_to_char

def load_shared_content(fh, shared_content):
    for line in fh:
        row = line.rstrip().split("\t")
        key = row[0]
        value = json.loads(row[1])
        shared_content[key] = value

def load_mnli_shared_content():
    shared_file_exist = False
    # shared_path = config.datapath + "/shared_2D_EM.json"
    # shared_path = config.datapath + "/shared_anto.json"
    # shared_path = config.datapath + "/shared_NER.json"
    shared_path = config.datapath + "/shared.jsonl"
    # shared_path = "../shared.json"
    print(shared_path)
    if os.path.isfile(shared_path):
        shared_file_exist = True
    # shared_content = {}
    assert shared_file_exist
    # if not shared_file_exist and config.use_exact_match_feature:
    #     with open(shared_path, 'w') as f:
    #         json.dump(dict(reconvert_shared_content), f)
    # elif config.use_exact_match_feature:
    with open(shared_path) as f:
        shared_content = {}
        load_shared_content(f, shared_content)
        # shared_content = json.load(f)
    return shared_content
def save_submission(path, ids, pred_ids):
    assert(ids.shape[0] == pred_ids.shape[0])
    reverse_label_map = {str(value): key for key, value in LABEL_MAP.items()}
    f = open(path, 'w')
    f.write("pairID,gold_label\n")
    for i in range(ids.shape[0]):
        pred = pred_ids[i]
        f.write("{},{}\n".format(str(ids[i]), reverse_label_map[str(pred)]))
        # f.write("{},{}\n".format(str(ids[i]), str(pred)))
    f.close()
    
def loadEmbedding_rand(path, word_indices, divident = 1.0): # TODO double embedding
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    j = 0
    n = len(word_indices)
    m = FIXED_PARAMETERS["word_embedding_dim"]
    emb = np.empty((n, m), dtype=np.float32)

    emb[:,:] = np.random.normal(size=(n,m)) / divident

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0, :] = np.zeros((1,m), dtype="float32")
    
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if FIXED_PARAMETERS["embeddings_to_load"] != None:
                if i >= FIXED_PARAMETERS["embeddings_to_load"]:
                    break
            
            s = line.split()
            if s[0] in word_indices:
                try:
                    emb[word_indices[s[0]], :] = np.asarray(s[1:])
                except ValueError:
                    print(s[0])
                    continue

    return emb

def is_synonym(w1, w2):
    for w1_lemma in w1.lemma_names():
        if w1_lemma in w2.lemma_names():
            return 1
    return 0

def is_antonym(w1, w2):
    for w1_lemma in w1.lemmas():
        for w1_anto_lemma in w1_lemma.antonyms():
            if w1_anto_lemma.name() in w2.lemma_names():
                # print(w1_syn.definition(), w2_syn.definition())
                return 1
    return 0

def is_hypernym(w1, w2, depth):
    for w in w1:
        tmp = w.hypernyms()
        if tmp is not None:
            if w2 in tmp:
                return depth
            else:
                return is_hypernym(tmp, w2, depth+1)
    return 0

def is_hyponym(w1, w2, depth):
    for w in w2:
        tmp = w.hyponyms()
        if tmp is not None:
            if w1 in tmp:
                return depth
            else:
                return is_hyponym(w1, tmp, depth+1)
    return 0

def is_co_hypernym(w1, w2):
    tmp = w2.lowest_common_hypernyms(w1)
    if tmp is not None:
        return 1
    return 0



