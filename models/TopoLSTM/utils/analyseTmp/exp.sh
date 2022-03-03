python -u gen_cas.py --rawdataset=dataset_dblp.txt --dataset=dblp1/ --observation_time=6 --least_num=10 --up_num=1000
python -u gen_run.py --rawdataset=dataset_dblp.txt --dataset=dblp1/ --observation_time=6 --least_num=10 --up_num=1000
python -u run.py --rawdataset=dataset_dblp.txt --dataset=dblp1/ --observation_time=6 --least_num=10 --up_num=1000 > 6.log 
