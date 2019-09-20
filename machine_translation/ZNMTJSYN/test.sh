config_file=/data/wjiang/nmt-exp/ch-en-syn-model/config.cfg
parser_config_file=./parser.cfg
best_step=325686
src_file=/data/wjiang/data/NMT/ch-en-mt/nist05.cn
tgt_file=/data/wjiang/data/NMT/ch-en-mt/nist05.en
gpu=3
python -u driver/test.py --config_file=$config_file --best_step=$best_step --src_file=$src_file --tgt_file=$tgt_file --gpu=$gpu --parser_config_file=$parser_config_file
