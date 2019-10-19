config_file=/data/LaGroup/wjiang/nmt-exp/mtl-shared-encoder-bpe/config.cfg
best_step=486778
src_file=/data/LaGroup/wjiang/data/NMT/ch-en-mt/nist06-bpe.cn
tgt_file=/data/LaGroup/wjiang/data/NMT/ch-en-mt/nist06.en
gpu=3
python -u driver/test.py --config_file=$config_file --best_step=$best_step --src_file=$src_file --tgt_file=$tgt_file --gpu=$gpu

