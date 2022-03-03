# python -u analyse.py --rawdataset=dataset_weibo.txt --dataset=weibo1/ --observation_time=3600 --interval=180 --up_num=100

# python -u analyse.py --rawdataset=dataset_weibo.txt --dataset=weibo2/ --observation_time=7200 --interval=180 --up_num=100

# python -u analyse.py --rawdataset=dataset_weibo.txt --dataset=weibo3/ --observation_time=10800 --interval=180 --up_num=100

python -u analyse.py --rawdataset=dataset_dblp.txt --dataset=dblp1/ --observation_time=6 --interval=1 --n_time_interval=6 --least_num=10 --up_num=100

python -u analyse.py --rawdataset=dataset_dblp.txt --dataset=dblp2/ --observation_time=7 --interval=1 --n_time_interval=7 --least_num=10 --up_num=100

python -u analyse.py --rawdataset=dataset_dblp.txt --dataset=dblp3/ --observation_time=8 --interval=1 --n_time_interval=8 --least_num=10 --up_num=100