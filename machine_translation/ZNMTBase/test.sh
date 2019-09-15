config_file=/data/wjiang/nmt-exp/en-de-model2/config.cfg
best_step=294168
src_file=/data/wjiang/data/NMT/ch-en-mt/nist06.cn
tgt_file=/data/wjiang/data/NMT/ch-en-mt/nist06.en
gpu=2
python -u driver/test.py --config_file=$config_file --best_step=$best_step --src_file=$src_file --tgt_file=$tgt_file --gpu=$gpu

