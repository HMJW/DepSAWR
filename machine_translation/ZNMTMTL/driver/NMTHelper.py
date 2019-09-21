import torch.nn.functional as F
from torch.autograd import Variable
from data.DataLoader import *
from module.Utils import *
from data.Vocab import NMTVocab
from torch.nn.utils.rnn import pad_sequence


class NMTHelper(object):
    def __init__(self, model, critic, src_vocab, tgt_vocab, config):
        self.model = model
        self.critic = critic
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config
        self.dep_critic = torch.nn.CrossEntropyLoss(reduction="sum")

    def prepare_training_data(self, src_inputs, tgt_inputs, src_trees):
        self.train_data = []
        #for idx in range(self.config.max_train_length):
        self.train_data.append([])
        for src_input, tgt_input, tree in zip(src_inputs, tgt_inputs, src_trees):
            #idx = int(len(src_input) - 1)
            self.train_data[0].append((self.src_data_id(src_input), self.tgt_data_id(tgt_input), self.arc_data_id(tree), self.rel_data_id(tree)))
        self.train_size = len(src_inputs)
        self.batch_size = self.config.train_batch_size
        batch_num = 0
        #for idx in range(self.config.max_train_length):
        train_size = len(self.train_data[0])
        batch_num += int(np.ceil(train_size / float(self.batch_size)))
        self.batch_num = batch_num

    def prepare_valid_data(self, src_inputs, tgt_inputs):
        self.valid_data = []
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            self.valid_data.append((self.src_data_id(src_input), self.tgt_data_id(tgt_input)))
        self.valid_size = len(self.valid_data)

    def src_data_id(self, src_input):
        result = self.src_vocab.word2id(src_input)
        return [self.src_vocab.BOS] + result + [self.src_vocab.EOS]

    def tgt_data_id(self, tgt_input):
        result = self.tgt_vocab.word2id(tgt_input)
        return [self.tgt_vocab.BOS] + result + [self.tgt_vocab.EOS]

    def arc_data_id(self, src_tree):
        result = [d.head for d in src_tree]
        return [-1] + result + [-1]

    def rel_data_id(self, src_tree):
        result = self.src_vocab.rel2id([d.rel for d in src_tree])
        return [self.src_vocab.BOS] + result + [self.tgt_vocab.EOS]


    def prepare_eval_data(self, src_inputs):
        eval_data = []
        for src_input in src_inputs:
            eval_data.append((self.src_data_id(src_input), src_input))

        return eval_data

    def pair_data_variable(self, batch, training=True):
        batch_size = len(batch)

        src_lengths = [len(batch[i][0]) for i in range(batch_size)]
        max_src_length = int(np.max(src_lengths))

        tgt_lengths = [len(batch[i][1]) for i in range(batch_size)]
        max_tgt_length = int(np.max(tgt_lengths))
        if not training:
            src_words = Variable(torch.LongTensor(batch_size, max_src_length).fill_(NMTVocab.PAD), requires_grad=False)
            tgt_words = Variable(torch.LongTensor(batch_size, max_tgt_length).fill_(NMTVocab.PAD), requires_grad=False)

            for b, instance in enumerate(batch):
                for index, word in enumerate(instance[0]):
                    src_words[b, index] = word
                for index, word in enumerate(instance[1]):
                    tgt_words[b, index] = word
                b += 1

            if self.use_cuda:
                src_words = src_words.cuda(self.device)
                tgt_words = tgt_words.cuda(self.device)

            return src_words, tgt_words, src_lengths, tgt_lengths
        else:
            src_words = Variable(torch.LongTensor(batch_size, max_src_length).fill_(NMTVocab.PAD), requires_grad=False)
            src_arcs = Variable(torch.LongTensor(batch_size, max_src_length).fill_(-1), requires_grad=False)
            src_rels = Variable(torch.LongTensor(batch_size, max_src_length).fill_(-1), requires_grad=False)
            tgt_words = Variable(torch.LongTensor(batch_size, max_tgt_length).fill_(NMTVocab.PAD), requires_grad=False)

            for b, instance in enumerate(batch):
                for index, word in enumerate(instance[0]):
                    src_words[b, index] = word
                for index, word in enumerate(instance[1]):
                    tgt_words[b, index] = word
                for index, word in enumerate(instance[2]):
                    src_arcs[b, index] = word
                for index, word in enumerate(instance[3]):
                    src_rels[b, index] = word
                b += 1

            if self.use_cuda:
                src_words = src_words.cuda(self.device)
                src_arcs = src_arcs.cuda(self.device)
                src_rels = src_rels.cuda(self.device)
                tgt_words = tgt_words.cuda(self.device)

            return src_words, src_arcs, src_rels, tgt_words, src_lengths, tgt_lengths

    def source_data_variable(self, batch):
        batch_size = len(batch)
        src_lengths = [len(batch[i][0]) for i in range(batch_size)]
        max_src_length = int(src_lengths[0])

        src_words = Variable(torch.LongTensor(batch_size, max_src_length).fill_(NMTVocab.PAD), requires_grad=False)
        for b, instance in enumerate(batch):
            for index, word in enumerate(instance[0]):
                src_words[b, index] = word
            b += 1

        if self.use_cuda:
            src_words = src_words.cuda(self.device)
        return src_words, src_lengths

    def compute_forward(self, seqs_x, src_arcs, src_rels, seqs_y, xlengths, normalization=1.0):
        """
        :type model: Transformer

        :type critic: NMTCritierion
        """

        y_inp = seqs_y[:, :-1].contiguous()
        y_label = seqs_y[:, 1:].contiguous()

        s_src, s_rel, dep_feature = self.model.forward_dep(seqs_x, "train")
        dep_loss = self.compute_dep_loss(s_src, s_rel, src_arcs, src_rels)
        dec_outs = self.model(seqs_x, dep_feature, y_inp, "train", lengths=xlengths)

        loss = self.critic(generator=self.model.generator,
                      normalization=normalization,
                      dec_outs=dec_outs,
                      labels=y_label)
        loss /= dec_outs.size(0)
        dep_loss /= dec_outs.size(0)
        mask = y_label.data.ne(NMTVocab.PAD)
        pred = self.model.generator(dec_outs).data.max(2)[1]  # [batch_size, seq_len]

        num_correct = y_label.data.eq(pred).float().masked_select(mask).sum() / normalization
        num_total = mask.sum().float()

        stats = Statistics(loss.item(), dep_loss.item(), num_total, num_correct)
        return loss + dep_loss, stats
    
    def compute_dep_loss(self, s_src, s_rel, gold_src, gold_rel):
        mask = gold_src.ne(-1)
        mask[:, 0] = 1
        s_src.masked_fill_(~mask.unsqueeze(1), -10000)
        mask[:, 0] = 0
        s_arc, s_rel = s_src[mask], s_rel[mask]
        gold_src, gold_rel = gold_src[mask], gold_rel[mask]
        s_rel = s_rel[torch.arange(len(s_rel)), gold_src]

        arc_loss = self.dep_critic(s_arc, gold_src)
        rel_loss = self.dep_critic(s_rel, gold_rel)
        loss = arc_loss + rel_loss
        return loss


    def train_one_batch(self, batch):
        self.model.train()
        self.model.zero_grad()
        src_words, src_arcs, src_rels, tgt_words, src_lengths, tgt_lengths = self.pair_data_variable(batch)
        loss, stat = self.compute_forward(src_words, src_arcs, src_rels, tgt_words, src_lengths)
        loss = loss / self.config.update_every
        loss.backward()
        return stat

    def valid(self, global_step):
        valid_stat = Statistics()
        self.model.eval()
        for batch in create_batch_iter(self.valid_data, self.config.test_batch_size):
            src_words, tgt_words, src_lengths, tgt_lengths = self.pair_data_variable(batch, False)
            loss, stat = self.compute_forward(src_words, tgt_words, src_lengths)
            valid_stat.update(stat)
        valid_stat.print_valid(global_step)
        return valid_stat


    def translate(self, eval_data):
        self.model.eval()
        result = {}
        for batch in create_batch_iter(eval_data, self.config.test_batch_size):
            batch_size = len(batch)
            src_words, src_lengths = self.source_data_variable(batch)
            allHyp = self.translate_batch(src_words, src_lengths)
            all_hyp_inds = [beam_result[0] for beam_result in allHyp]
            for idx in range(batch_size):
                if all_hyp_inds[idx][-1] == self.tgt_vocab.EOS:
                    all_hyp_inds[idx].pop()
            all_hyp_words = [self.tgt_vocab.id2word(idxs) for idxs in all_hyp_inds]
            for idx, instance in enumerate(batch):
                result['\t'.join(instance[1])] = all_hyp_words[idx]

        return result


    def translate_batch(self, src_inputs, src_input_lengths):
        _, _, dep_feature = self.model.forward_dep(src_inputs)
        word_ids = self.model(src_inputs, dep_feature, lengths=src_input_lengths, mode="infer", beam_size=self.config.beam_size)
        word_ids = word_ids.cpu().numpy().tolist()

        result = []
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != NMTVocab.PAD] for line in sent_t]
            result.append(sent_t)

        return result
