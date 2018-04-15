import json, codecs, nltk, sys, time, operator, os
import numpy as np
from collections import OrderedDict, Counter

# dir_path = '/home/hongmin/diskhome/qangaroo/data/qangaroo_v1.1/wikihop/statistics/'

class Vocab(object):
    def __init__(self, train_set, emb_path=None):

        self.dataset = train_set
        self.emb_path = emb_path
        self.dir_path = os.path.dirname(self.dataset.input_path)
        self.PAD = 0
        self.UNK = 1

        self.vocab_counter = Counter()
        self.vocab_list = [('<PAD>', 0), ('<UNK>', 0)]
        self.vocab_size = 0
        self.vocab_size_pretrain = 0
        self.emb_dim = 0

        self.all_query_relations = None

        self.pretrain_embeddings = None
        self.pretrain_embeddings_lookup = dict()

        self.word2idx = None
        self.idx2word = None
        self.pre2idx = dict()
        self.idx2pre = dict()

        self.save_query_rels()
        if self.emb_path:
            print('Loading pre-trained embeddings ...')
            self.load_pre_emb()
        else:
            self.process_vocab()


    def process_vocab(self):

        # update vocab_counter to include all tokens in query entities, relations, and supporting documents
        for doc in self.dataset.get_all_docs():
            self.vocab_counter.update(doc.split())
        for ent in self.dataset.get_all_query_ents():
            self.vocab_counter.update(ent.split())
        for rel in self.dataset.get_all_query_rels():
            self.vocab_counter.update(rel.split('_'))

        # save vocab file
        self.save_count_dict(os.path.join(self.dir_path, 'train.vocab'), self.vocab_counter)

        self.vocab_list = self.vocab_list + self.vocab_counter.most_common()
        self.vocab_size = len(self.vocab_list)

        self.word2idx = {word: idx for idx, (word, count) in enumerate(self.vocab_list)}
        self.idx2word = {idx: word for idx, (word, count) in enumerate(self.vocab_list)}

    def save_query_rels(self):
        self.all_query_relations = Counter(self.dataset.get_all_query_rels())
        self.save_count_dict(os.path.join(self.dir_path, 'train.query_rels'), self.all_query_relations)

    def load_pre_emb(self):
        vectors = []
        with codecs.open(self.emb_path, 'r', encoding="utf-8") as fin:
            for idx, line in enumerate(fin):
                values = line.split()
                word = values[0]
                self.pre2idx[word] = idx+1
                self.idx2pre[idx+1] = word
                temp = values[1:]
                vectors.append(temp)
                self.pretrain_embeddings_lookup[word] = np.array(temp, dtype=np.float32)

        vectors = np.array(vectors, dtype=np.float32)
        # vectors /= np.std(vectors)
        pad_emb = np.zeros((1, vectors.shape[1]), dtype=np.float32)
        unk_emb = np.mean(vectors, axis=0, keepdims=True)
        special_emb = np.concatenate((pad_emb, unk_emb), axis=0)
        self.pretrain_embeddings = np.concatenate((special_emb, vectors), axis=0)
        self.vocab_size_pretrain, self.emb_dim = self.pretrain_embeddings.shape

    def save_count_dict(self, filename, output_dict):
        with codecs.open(filename + '.sort_key.json', 'w+', encoding="utf-8") as fout:
            json.dump(output_dict, fout, indent=1, ensure_ascii=False, sort_keys=True)
        with codecs.open(filename + '.sort_count.json', 'w+', encoding="utf-8") as fout:
            output_dict = OrderedDict(sorted(output_dict.items(), key=operator.itemgetter(1), reverse=True))
            json.dump(output_dict, fout, indent=1, ensure_ascii=False, sort_keys=False)

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab_size_pretrain(self):
        return self.vocab_size_pretrain

    def get_index(self, word):
        return self.word2idx[word] if word in self.word2idx else self.UNK

    def get_index_pretrain(self, word):
        return self.pre2idx[word] if word in self.pre2idx else self.UNK


