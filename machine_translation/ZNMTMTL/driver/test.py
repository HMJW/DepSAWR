# -*- coding: utf-8 -*-
import sys
sys.path.extend(["../","./"])
import argparse
import random
from driver.Config import *
from driver.Optim import *
from driver.NMTHelper import *
from model.DL4MT import DL4MT
from model.Transformer import Transformer
from module.Criterions import *
import pickle
import os
import re


def evaluate(nmt, src_file, tgt_files, config, global_step):
    valid_srcs = read_corpus(src_file)
    eval_data = nmt.prepare_eval_data(valid_srcs)
    result = nmt.translate(eval_data)

    outputFile = src_file + '.' + str(global_step)
    output = open(outputFile, 'w', encoding='utf-8')
    ordered_result = []
    for idx, instance in enumerate(eval_data):
        src_key = '\t'.join(instance[1])
        cur_result = result.get(src_key)
        if cur_result is not None:
            ordered_result.append(cur_result)
        else:
            print("Strange, miss one sentence")
            ordered_result.append([''])

        sentence_out = ' '.join(ordered_result[idx])
        sentence_out = sentence_out.replace(' <unk>', '')
        sentence_out = sentence_out.replace('@@ ', '')

        output.write(sentence_out + '\n')

    output.close()

    command = 'perl %s %s < %s' % (config.bleu_script, ' '.join(tgt_files), outputFile)
    bleu_exec = os.popen(command)
    bleu_exec = bleu_exec.read()
    # Get bleu value
    bleu_val = re.findall('BLEU = (.*?),', bleu_exec, re.S)[0]
    bleu_val = float(bleu_val)
    try:
        os.remove(outputFile)
    except Exception as e:
        pass
    return bleu_val



if __name__ == '__main__':
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)


    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='/data/wjiang/nmt-exp/en-de-model3/config.cfg')
    argparser.add_argument('--best_step', type=int, default=168096)
    argparser.add_argument('--src_file', default="/data/wjiang/data/NMT/ch-en-mt/nist03.cn")
    argparser.add_argument('--tgt_file', default="/data/wjiang/data/NMT/ch-en-mt/nist03.en")
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--gpu', default=-1, type=int, help='Use id of gpu, -1 if cpu.')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.gpu >= 0:
        config.use_cuda = True
        torch.cuda.set_device(args.gpu)
        print("GPU ID: ", args.gpu)
    print("\nGPU using status: ", config.use_cuda)

    print(args.tgt_file)
    print("loading vocab...")
    src_vocab = pickle.load(open(config.load_src_vocab_path, "rb"))
    tgt_vocab = pickle.load(open(config.load_tgt_vocab_path, "rb"))

    print(f"loading best model at {args.best_step} step...")
    nmt_model = eval(config.model_name)(config, src_vocab.size, tgt_vocab.size, config.use_cuda)
    model_path = config.load_model_path + f".{args.best_step}"
    nmt_model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    critic = NMTCritierion(label_smoothing=config.label_smoothing)
    tgt_files = [args.tgt_file+str(i) for i in range(4)]

    if config.use_cuda:
        #torch.backends.cudnn.enabled = False
        nmt_model = nmt_model.cuda()
        critic = critic.cuda()

    nmt = NMTHelper(nmt_model, critic, src_vocab, tgt_vocab, config)
    bleu = evaluate(nmt, args.src_file, tgt_files, config, args.best_step)
    print(f"bleu : {bleu}")