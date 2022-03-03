# python -u gen_cas.py --rawdataset=dataset_weibo.txt --dataset=weibo1/ --observation_time=3600 --interval=180
# python -u gen_graph_signal.py --rawdataset=dataset_weibo.txt --dataset=weibo1/ --observation_time=3600 --interval=180
python -u run.py --rawdataset=dataset_weibo.txt --dataset=weibo1/ --observation_time=3600 --interval=180 > 3600*1.log

# python -u gen_cas.py --rawdataset=dataset_weibo.txt --dataset=weibo2/ --observation_time=7200 --interval=180
# python -u gen_graph_signal.py --rawdataset=dataset_weibo.txt --dataset=weibo2/ --observation_time=7200 --interval=180
python -u run.py --rawdataset=dataset_weibo.txt --dataset=weibo2/ --observation_time=7200 --interval=180 > 3600*2.log

python -u gen_cas.py --rawdataset=dataset_weibo.txt --dataset=weibo3/ --observation_time=10800 --interval=180
python -u gen_graph_signal.py --rawdataset=dataset_weibo.txt --dataset=weibo3/ --observation_time=10800 --interval=180
python -u run.py --rawdataset=dataset_weibo.txt --dataset=weibo3/ --observation_time=10800 --interval=180 > 3600*3.log


# python -u gen_cas.py --rawdataset=dataset_weibo.txt --dataset=weibolong/ --observation_time=3600 --interval=180
# python -u gen_graph_signal.py --rawdataset=dataset_weibo.txt --dataset=weibolong/ --observation_time=3600 --interval=180
# python -u run.py --rawdataset=dataset_weibo.txt --dataset=weibolong/ --observation_time=3600 --interval=180 > 3600*1long.log