# python -u gen_cas.py --rawdataset=dataset_dblp.txt --dataset=dblp1/ --observation_time=6 --interval=1 --least_num=10
# python -u gen_graph_signal.py --rawdataset=dataset_dblp.txt --dataset=dblp1/ --observation_time=6 --interval=1 --least_num=10
# python -u run.py --rawdataset=dataset_dblp.txt --dataset=dblp1/ --observation_time=6 --interval=1 --least_num=10 > 6.log

# python -u gen_cas.py --rawdataset=dataset_dblp.txt --dataset=dblp2/ --observation_time=7 --interval=1 --least_num=10
# python -u gen_graph_signal.py --rawdataset=dataset_dblp.txt --dataset=dblp2/ --observation_time=7 --interval=1 --least_num=10
# python -u run.py --rawdataset=dataset_dblp.txt --dataset=dblp2/ --observation_time=7 --interval=1 --least_num=10 > 7.log

# python -u gen_cas.py --rawdataset=dataset_dblp.txt --dataset=dblp3/ --observation_time=8 --interval=1 --least_num=10
# python -u gen_graph_signal.py --rawdataset=dataset_dblp.txt --dataset=dblp3/ --observation_time=8 --interval=1 --least_num=10
nohup python -u run.py --rawdataset=dataset_dblp.txt --dataset=dblp3/ --observation_time=8 --interval=1 --least_num=10 > 8.log
