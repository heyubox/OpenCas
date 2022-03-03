python -u analyse.py --rawdataset=dataset_weibo.txt --dataset=weibo1/ --observation_time=3600 --up_num=100
python -u analyse.py --rawdataset=dataset_weibo.txt --dataset=weibo2/ --observation_time=7200 --up_num=100
python -u analyse.py --rawdataset=dataset_weibo.txt --dataset=weibo3/ --observation_time=10800 --up_num=100

python -u analyse.py --rawdataset=dataset_dblp.txt --dataset=dblp1/ --observation_time=6 --least_num=10 --up_num=100
python -u analyse.py  --rawdataset=dataset_dblp.txt --dataset=dblp2/ --observation_time=7 --least_num=10 --up_num=100
python -u analyse.py --rawdataset=dataset_dblp.txt --dataset=dblp3/ --observation_time=8 --least_num=10 --up_num=100