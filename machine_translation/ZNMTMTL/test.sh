config_file=/data2/wjiang/nmt-exp/mtl/config.cfg
best_step=357204
src_file=/data2/wjiang/data/NMT/ch-en-mt/nist06.cn
tgt_file=/data2/wjiang/data/NMT/ch-en-mt/nist06.en
gpu=5
python -u driver/test.py --config_file=$config_file --best_step=$best_step --src_file=$src_file --tgt_file=$tgt_file --gpu=$gpu

