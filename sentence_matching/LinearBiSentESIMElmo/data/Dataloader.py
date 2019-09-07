from data.Vocab import *
from data.Instance import *
from data.TensorInstances import *
from collections import Counter
import codecs

def read_corpus(file):
    data = []
    with codecs.open(file, encoding='utf8') as input_file:
        curtext = []
        for line in input_file.readlines():
            line = line.strip()
            if line is not None and line != '':
                curtext.append(line)
            else:
                slen = len(curtext)
                if slen == 3:
                    cur_data = parseInstance(curtext)
                    if cur_data.src_len <= 500 and cur_data.tgt_len <=500:
                        data.append(cur_data)
                curtext = []

    slen = len(curtext)
    if slen == 3:
        cur_data = parseInstance(curtext)
        if cur_data.src_len <= 500 and cur_data.tgt_len <= 500:
            data.append(cur_data)

    print("Total num: " + str(len(data)))
    return data

def creatVocab(corpusFile, min_occur_count):
    word_counter = Counter()
    action_counter = Counter()
    tag_counter = Counter()
    alldatas = read_corpus(corpusFile)
    for inst in alldatas:
        for curword in inst.src_forms:
            word_counter[curword] += 1
            items = curword.split('##')
            if len(items) == 3 and (items[0] == 'arc'or items[0] == 'pop'):
                action_counter[curword] += 1
        for curword in inst.tgt_forms:
            word_counter[curword] += 1
            items = curword.split('##')
            if len(items) == 3 and (items[0] == 'arc'or items[0] == 'pop'):
                action_counter[curword] += 1
        tag_counter[inst.tag] += 1

    return Vocab(word_counter, action_counter, tag_counter, min_occur_count)

def insts_numberize(insts, vocab):
    for inst in insts:
        yield inst2id(inst, vocab)

def inst2id(inst, vocab):
    src_ids = vocab.word2id(inst.src_forms)
    tgt_ids = vocab.word2id(inst.tgt_forms)
    src_acids = vocab.action2id(inst.src_forms)
    tgt_acids = vocab.action2id(inst.tgt_forms)

    tagid = vocab.tag2id(inst.tag)

    return src_ids, src_acids, tgt_ids, tgt_acids, tagid, inst


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        insts = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield insts


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  insts in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def batch_data_variable(batch, vocab):
    slen, tlen = len(batch[0].src_words), len(batch[0].tgt_words)
    batch_size = len(batch)
    for b in range(1, batch_size):
        cur_slen, cur_tlen = len(batch[b].src_words), len(batch[b].tgt_words)
        if cur_slen > slen: slen = cur_slen
        if cur_tlen > tlen: tlen = cur_tlen

    tinst = TensorInstances(batch_size, slen, tlen)
    src_words, tgt_words = [], []
    src_word_indexes, tgt_word_indexes = [], []

    b = 0
    for src_ids, src_acids, tgt_ids, tgt_acids, tagid, inst in insts_numberize(batch, vocab):
        tinst.tags[b] = tagid
        cur_slen, cur_tlen = len(src_ids), len(tgt_ids)
        tinst.src_lens[b] = cur_slen
        tinst.tgt_lens[b] = cur_tlen

        for index in range(cur_slen):
            tinst.src_words[b, index] = src_ids[index]
            tinst.src_actions[b, index] = src_acids[index]
            tinst.src_masks[b, index] = 1
        for index in range(cur_tlen):
            tinst.tgt_words[b, index] = tgt_ids[index]
            tinst.tgt_actions[b, index] = tgt_acids[index]
            tinst.tgt_masks[b, index] = 1

        cur_src_words, cur_src_indexes = [], []
        for index, curword in enumerate(inst.src_words):
            items = curword.split('##')
            if len(items) != 3 or (items[0] != 'arc' and items[0] != 'pop'):
                cur_src_words.append(curword)
                cur_src_indexes.append(index)

        cur_tgt_words, cur_tgt_indexes = [], []
        for index, curword in enumerate(inst.tgt_words):
            items = curword.split('##')
            if len(items) != 3 or (items[0] != 'arc' and items[0] != 'pop'):
                cur_tgt_words.append(curword)
                cur_tgt_indexes.append(index)

        src_words.append(cur_src_words)
        tgt_words.append(cur_tgt_words)
        src_word_indexes.append(cur_src_indexes)
        tgt_word_indexes.append(cur_tgt_indexes)

        b += 1
    return tinst, src_words, src_word_indexes, tgt_words, tgt_word_indexes

def batch_variable_inst(insts, tagids, vocab):
    for inst, tagid in zip(insts, tagids):
        pred_tag = vocab.id2tag(tagid)
        yield Instance(inst.src_words, inst.tgt_words, pred_tag), pred_tag == inst.tag
