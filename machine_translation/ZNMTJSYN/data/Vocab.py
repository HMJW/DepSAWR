import numpy as np
import sys
import copy
sys.path.extend(["../","./"])

class NMTVocab:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    S_PAD, S_BOS, S_EOS, S_UNK = '<pad>', '<s>', '</s>', '<unk>'
    def __init__(self, word_list):
        """
        :param word_list: list of words
        """
        self.i2w = [self.S_PAD, self.S_BOS, self.S_EOS, self.S_UNK] + word_list

        reverse = lambda x: dict(zip(x, range(len(x))))
        self.w2i = reverse(self.i2w)
        if len(self.w2i) != len(self.i2w):
            print("serious bug: words dumplicated, please check!")

        print("Vocab info: #words %d" % (self.size))


    def word2id(self, xs):
        if isinstance(xs, list):
            return [self.w2i.get(x, self.UNK) for x in xs]
        return self.w2i.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self.i2w[x] for x in xs]
        return self.i2w[xs]

    def save2file(self, outfile):
        with open(outfile, 'w', encoding='utf8') as file:
            for id, word in enumerate(self.i2w):
                if id > self.UNK: file.write(word + '\n')
            file.close()

    @property
    def size(self):
        return len(self.i2w)


class Vocab(object):
    PAD, ROOT, UNK = 0, 1, 2
    def __init__(self, word_counter, chars, rel_counter, relroot='root', min_occur_count = 2):
        self._root = relroot
        self._root_form = '<' + relroot.lower() + '>'
        self._id2word = ['<pad>', self._root_form, '<unk>']
        self._wordid2freq = [10000, 10000, 10000]
        self._id2extword = ['<pad>', self._root_form, '<unk>']
        self._id2rel = ['<pad>', relroot]
        self._chars = ['<pad>', self._root_form, '<unk>'] + chars
        self._char2id = {c:i for i, c in enumerate(self._chars)}

        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)
                self._wordid2freq.append(count)

        for rel, count in rel_counter.most_common():
            if rel != relroot: self._id2rel.append(rel)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        self._rel2id = reverse(self._id2rel)
        if len(self._rel2id) != len(self._id2rel):
            print("serious bug: relation labels dumplicated, please check!")

        print("Vocab info: #words %d, #chars %d, #rels %d" % (self.vocab_size, self.char_size, self.rel_size))

    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                self._id2extword.append(values[0])
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1
        self._chars += sorted(set("".join(self._id2extword)).difference(self._char2id))
        self._char2id = {c:i for i, c in enumerate(self._chars)}

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        if len(self._extword2id) != len(self._id2extword):
            print("serious bug: extern words dumplicated, please check!")

        return embeddings

    def create_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword) - word_count
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if self._extword2id.get(values[0], self.UNK) != index:
                    print("Broken vocab or error embedding file, please check!")
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        embeddings = embeddings / np.std(embeddings)

        return embeddings


    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x.lower(), self.UNK) for x in xs]
        return self._word2id.get(xs.lower(), self.UNK)

    def char2id(self, word, max_length=20):
        ids = [self._char2id.get(c, self.UNK)
                                for c in word[:max_length]]
        ids += [0] * (max_length - len(word))
        return ids

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def wordid2freq(self, xs):
        if isinstance(xs, list):
            return [self._wordid2freq[x] for x in xs]
        return self._wordid2freq[xs]

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.UNK) for x in xs]
        return self._extword2id.get(xs, self.UNK)

    def id2extword(self, xs):
        if isinstance(xs, list):
            return [self._id2extword[x] for x in xs]
        return self._id2extword[xs]

    def rel2id(self, xs):
        if isinstance(xs, list):
            return [self._rel2id[x] for x in xs]
        return self._rel2id[xs]

    def id2rel(self, xs):
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]

    def copyfrom(self, vocab):
        self._root = vocab._root
        self._root_form = vocab._root_form

        self._id2word = copy.deepcopy(vocab._id2word)
        self._wordid2freq = copy.deepcopy(vocab._wordid2freq)
        self._id2extword = copy.deepcopy(vocab._id2extword)
        self._chars = copy.deepcopy(vocab._chars)
        self._id2rel = copy.deepcopy(vocab._id2rel)

        self._word2id = copy.deepcopy(vocab._word2id)
        self._extword2id = copy.deepcopy(vocab._extword2id)
        self._rel2id = copy.deepcopy(vocab._rel2id)
        self._char2id = copy.deepcopy(vocab._char2id)

        print("Vocab info: #words %d, #rels %d" % (self.vocab_size, self.rel_size))


    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def extvocab_size(self):
        return len(self._id2extword)

    @property
    def rel_size(self):
        return len(self._id2rel)

    @property
    def char_size(self):
        return len(self._char2id)