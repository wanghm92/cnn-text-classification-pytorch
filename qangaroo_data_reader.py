import json, codecs, sys, time, random
import numpy as np

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    # starting and ending point of each batch
    return [(i*batch_size, (i+1)*batch_size) for i in range(0, nb_batch)]

IDX, QUERY, ANSWER, CAND, DOCS, ANN = ['id', 'query', 'answer', 'candidates', 'supports', 'annotations']

class QAngarooDataReader(object):
    def __init__(self, input_path, args, is_test=False, has_ann=False):
        self.input_path = input_path
        self.has_ann = has_ann
        self.is_test = is_test
        self.vocab = None
        self.batch_size = args.batch_size
        self.is_shuffle = args.is_shuffle
        self.concat = args.concat

        self._all_items_raw = []   # raw data loaded from JSON file
        self._all_items_after = [] # processed raw data strings
        self._samples = []         # indexed data samples
        self._number_of_items = 0
        self._number_of_samples = 0
        self._batches = []
        self._number_of_batches = 0
        self._index_array = None
        self._cur_pointer = 0

        self._all_query_rels = []
        self._all_query_ents = []
        self._all_answers = []
        self._all_candidates = []
        self._all_docs = []

        self.load_data()

    def load_data(self):
        # read file
        with codecs.open(self.input_path, 'r', encoding="utf-8") as fin:
            self._all_items_raw = json.load(fin)

        # process samples
        max_query_length = 0
        for i, item in enumerate(self.get_all_items_raw()):

            query_tuple = item[QUERY].split()
            query_rel = query_tuple[0]  # string with '_' between words
            query_ent = ' '.join(query_tuple[1:]).lower()  # multi-word string

            query_length = len(query_ent.split()) + len(query_rel.split('_'))
            if query_length > max_query_length: max_query_length = query_length

            answer_ent = item[ANSWER].lower()  # string
            candidates = [x.lower() for x in item[CAND]]
            support_docs = [x.strip().lower() for x in item[DOCS]]

            sample_dict_after = {IDX     : item[IDX],
                                 QUERY   : [query_rel, query_ent],
                                 ANSWER  : answer_ent,
                                 CAND    : candidates,
                                 DOCS    : support_docs}
            if self.has_ann: sample_dict_after[ANN] = item[ANN]

            self._all_items_after.append(sample_dict_after)
            self._all_query_rels.append(query_rel)
            self._all_query_ents.append(query_ent)
            self._all_answers.append(answer_ent)
            self._all_candidates.extend(candidates)
            self._all_docs.extend(support_docs)

        # print(sorted([len(d.split()) for d in self.get_all_docs()], reverse=True)[1000:1100])
        self._number_of_items = len(self._all_items_raw)

    def index_samples(self, vocab):
        def get_yesno_label_for_doc(d, a): return 1 if a in d else 0

        # DEBUG
        ones, zeros = 0, 0

        positive_samples = []
        self.vocab = vocab
        start_time = time.time()
        for i, item in enumerate(self.get_all_items_after()):

            query_tokens = item[QUERY][0].split('_') + item[QUERY][1].split()
            query_indices = self.sent2idx(query_tokens, max_length=20) if not self.concat else None
            answer = item[ANSWER]

            label_sum = 0
            tuples = []
            for d in item[DOCS]:
                doc_tokens = d.split()
                if self.concat:
                    doc_tokens = query_tokens + doc_tokens
                doc_indices = self.sent2idx(doc_tokens)
                doc_label = get_yesno_label_for_doc(d, answer) # 0 or 1

                # DEBUG
                if doc_label == 0:
                    zeros += 1
                else:
                    ones +=1
                    positive_samples.append((query_indices, doc_indices, doc_label))

                label_sum += doc_label
                tuples.append((query_indices, doc_indices, doc_label))

            # only include samples with answers present in the supporting documents
            if label_sum > 0 and tuples: self._samples.extend(tuples)

        print('{} items indexed in {:.2f} secs'.format(self.get_dataset_size(), time.time()-start_time))
        # DEBUG
        total = ones+zeros
        print('BEFORE: #1 = {}({:.2f}%)\t-vs-\t#0 = {} ({:.2f}%)'.format(ones, 100.0*ones/total,
                                                                         zeros, 100.0*zeros/total))

        if not self.is_test:
            num_pos_to_be_added = zeros//ones-1
            for i in range(num_pos_to_be_added):
                self._samples.extend(positive_samples)
            # DEBUG
            ones += num_pos_to_be_added*len(positive_samples)
            total = len(self._samples)
            print('AFTER: #1 = {}({:.2f}%)\t-vs-\t#0 = {} ({:.2f}%)'.format(ones, 100.0*ones/total,
                                                                            zeros, 100.0*zeros/total))
        self.extend_to_multiples_of_batchsize()
        sys.stdout.flush()

    def extend_to_multiples_of_batchsize(self):
        if not self.is_test:
            datasize = len(self._samples)
            num_to_add = self.batch_size*(datasize//self.batch_size + 1) - datasize
            samples_to_add = [self._samples[idx] for idx in random.sample(range(datasize), num_to_add)]
            self._samples += samples_to_add

        self._number_of_samples = len(self._samples)

    def sent2idx(self, tokens, max_length=500):
        sent_ids = [self.vocab.get_index_pretrain(word) for word in tokens]
        # zero paddings to equal length
        if len(sent_ids) < max_length: sent_ids = sent_ids + [self.vocab.PAD] * (max_length - len(sent_ids))
        return sent_ids[:max_length]

    def split_batches(self):
        # shuffle samples since they are added in sequence as they appear in the dataset
        if self.is_shuffle: random.shuffle(self._samples)

        batch_indices = make_batches(self.get_number_of_samples(), self.batch_size)
        self._number_of_batches = len(batch_indices)
        self._index_array = np.arange(self.get_num_batch()) # idx of batches
        for start, end in batch_indices:
            batch = self._samples[start:end]
            doc_indices = [x[1] for x in batch]
            doc_labels = [x[2] for x in batch]
            if not self.concat:
                query_indices = [x[0] for x in batch]
                self._batches.append((np.array(query_indices, dtype=np.int64),
                                      np.array(doc_indices, dtype=np.int64),
                                      np.array(doc_labels, dtype=np.int64)))
            else:
                self._batches.append((None,
                                      np.array(doc_indices, dtype=np.int64),
                                      np.array(doc_labels, dtype=np.int64)))

    def next_batch(self):
        if self.get_cur_pointer() >= self.get_num_batch():
            self.reset_batch_pointer()
            if self.is_shuffle: np.random.shuffle(self._index_array)
        # print('{} '.format(self.index_array[self.cur_pointer]))
        cur_batch = self._batches[self._index_array[self._cur_pointer]]
        self._cur_pointer += 1
        return cur_batch

    def get_cur_pointer(self):
        return self._cur_pointer

    def reset_batch_pointer(self):
        self._cur_pointer = 0

    def get_num_batch(self):
        return self._number_of_batches

    def get_batch(self, i):
        if i >= self.get_num_batch(): return None
        return self._batches[i]

    def get_dataset_size(self):
        return self._number_of_items

    def get_number_of_samples(self):
        return self._number_of_samples

    def get_all_items_raw(self):
        print('Accessing %d items'%self.get_dataset_size())
        return self._all_items_raw

    def get_all_items_after(self):
        print('Accessing %d items with lowercase strings'%self.get_dataset_size())
        return self._all_items_after

    def get_all_samples(self):
        print('Accessing %d samples'%self.get_number_of_samples())
        return self._samples

    def get_item_raw(self, idx):
        return self._all_items_raw[idx]

    def get_item_after(self, idx):
        return self._all_items_after[idx]

    def get_all_query_rels(self):
        print('Accessing %d query_rels'%len(self._all_query_rels))
        return self._all_query_rels

    def get_all_query_ents(self):
        print('Accessing %d query_ents'%len(self._all_query_ents))
        return self._all_query_ents

    def get_all_answers(self):
        print('Accessing %d answers'%len(self._all_answers))
        return self._all_answers

    def get_all_candidates(self):
        print('Accessing %d candidates'%len(self._all_candidates))
        return self._all_candidates

    def get_all_docs(self):
        print('Accessing %d docs'%len(self._all_docs))
        return self._all_docs