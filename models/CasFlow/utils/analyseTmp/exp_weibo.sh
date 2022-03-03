python -u gen_cas.py --rawdataset=dataset_weibo.txt --dataset=weibo1/ --observation_time=3600 --up_num=100
python -u gen_emb.py --rawdataset=dataset_weibo.txt --dataset=weibo1/ --observation_time=3600 --up_num=100
python -u run.py --rawdataset=dataset_weibo.txt --dataset=weibo1/ --observation_time=3600 --up_num=100 > 3600*1.log &

python -u gen_cas.py --rawdataset=dataset_weibo.txt --dataset=weibo2/ --observation_time=7200 --up_num=100
python -u gen_emb.py --rawdataset=dataset_weibo.txt --dataset=weibo2/ --observation_time=7200 --up_num=100
python -u run.py --rawdataset=dataset_weibo.txt --dataset=weibo2/ --observation_time=7200 --up_num=100 > 3600*2.log &

python -u gen_cas.py --rawdataset=dataset_weibo.txt --dataset=weibo3/ --observation_time=10800 --up_num=100
python -u gen_emb.py --rawdataset=dataset_weibo.txt --dataset=weibo3/ --observation_time=10800 --up_num=100
python -u run.py --rawdataset=dataset_weibo.txt --dataset=weibo3/ --observation_time=10800 --up_num=100 > 3600*3.log &

# python -u analyse.py --rawdataset=dataset_weibo.txt --dataset=weibo1/ --observation_time=3600
# python -u analyse.py --rawdataset=dataset_weibo.txt --dataset=weibo2/ --observation_time=7200
# python -u analyse.py --rawdataset=dataset_weibo.txt --dataset=weibo3/ --observation_time=10800

