import sys
sys.path.extend(["../","./"])

class NMTVocab:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    S_PAD, S_BOS, S_EOS, S_UNK = '<pad>', '<s>', '</s>', '<unk>'
    def __init__(self, word_list, rel_list=None):
        """
        :param word_list: list of words
        """
        self.i2w = [self.S_PAD, self.S_BOS, self.S_EOS, self.S_UNK] + word_list
       

        reverse = lambda x: dict(zip(x, range(len(x))))
        self.w2i = reverse(self.i2w)
        if rel_list:
            self.i2r = [self.S_PAD, self.S_BOS] + rel_list
            self.r2i = reverse(self.i2r)
        else:
            self.i2r, self.r2i = None, None
        if len(self.w2i) != len(self.i2w):
            print("serious bug: words dumplicated, please check!")

        if self.i2r:
            print("Vocab info: #words %d, #rels %d" % (self.size, self.rel_size))
        else:
            print("Vocab info: #words %d" % (self.size))

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self.w2i.get(x, self.UNK) for x in xs]
        return self.w2i.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self.i2w[x] for x in xs]
        return self.i2w[xs]
    
    def rel2id(self, xs):
        if isinstance(xs, list):
            return [self.r2i[x] for x in xs]
        return self.r2i[xs]

    def id2rel(self, xs):
        if isinstance(xs, list):
            return [self.i2r[x] for x in xs]
        return self.i2r[xs]

    def save2file(self, outfile):
        with open(outfile, 'w', encoding='utf8') as file:
            for id, word in enumerate(self.i2w):
                if id > self.UNK: file.write(word + '\n')
            file.close()

    @property
    def size(self):
        return len(self.i2w)

    @property
    def rel_size(self):
        return len(self.i2r)