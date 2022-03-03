# python -u gen_cas.py --rawdataset=dataset_weibo.txt --dataset=weibo1_long/ --observation_time=3600 --up_num=1000
# python -u gen_emb.py --rawdataset=dataset_weibo.txt --dataset=weibo1_long/ --observation_time=3600 --up_num=1000
python -u run.py --rawdataset=dataset_weibo.txt --dataset=weibo1_long/ --observation_time=3600 --up_num=1000 > 3600*1_long.log 

# python -u gen_cas.py --rawdataset=dataset_weibo.txt --dataset=weibo2_long/ --observation_time=7200 --up_num=1000
# python -u gen_emb.py --rawdataset=dataset_weibo.txt --dataset=weibo2_long/ --observation_time=7200 --up_num=1000
python -u run.py --rawdataset=dataset_weibo.txt --dataset=weibo2_long/ --observation_time=7200 --up_num=1000 > 3600*2_long.log 

# python -u gen_cas.py --rawdataset=dataset_weibo.txt --dataset=weibo3_long/ --observation_time=10800 --up_num=1000
# python -u gen_emb.py --rawdataset=dataset_weibo.txt --dataset=weibo3_long/ --observation_time=10800 --up_num=1000
python -u run.py --rawdataset=dataset_weibo.txt --dataset=weibo3_long/ --observation_time=10800 --up_num=1000 > 3600*3_long.log 

python -u analyse.py --rawdataset=dataset_weibo.txt --dataset=weibo1_long/ --observation_time=3600 > 3600*1_long.log
python -u analyse.py --rawdataset=dataset_weibo.txt --dataset=weibo2_long/ --observation_time=7200 > 3600*2_long.log 
python -u analyse.py --rawdataset=dataset_weibo.txt --dataset=weibo3_long/ --observation_time=10800 > 3600*3_long.log 

