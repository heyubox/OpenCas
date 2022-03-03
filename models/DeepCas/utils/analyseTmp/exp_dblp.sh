python -u gen_cas.py --rawdataset=dataset_dblp.txt --dataset=dblp1/ --observation_time=3 --least_num=10 --up_num=100
python -u gen_walks.py --rawdataset=dataset_dblp.txt --dataset=dblp1/ --observation_time=3 --least_num=10 --up_num=100
python -u gen_run.py --rawdataset=dataset_dblp.txt --dataset=dblp1/ --observation_time=3 --least_num=10 --up_num=100
python -u run.py --rawdataset=dataset_dblp.txt --dataset=dblp1/ --observation_time=3 --least_num=10 --up_num=100 > 3.log 

python -u gen_cas.py --rawdataset=dataset_dblp.txt --dataset=dblp2/ --observation_time=5 --least_num=10 --up_num=100
python -u gen_walks.py --rawdataset=dataset_dblp.txt --dataset=dblp2/ --observation_time=5 --least_num=10 --up_num=100
python -u gen_run.py --rawdataset=dataset_dblp.txt --dataset=dblp2/ --observation_time=5 --least_num=10 --up_num=100
python -u run.py --rawdataset=dataset_dblp.txt --dataset=dblp2/ --observation_time=5 --least_num=10 --up_num=100 > f.log 

python -u gen_cas.py --rawdataset=dataset_dblp.txt --dataset=dblp3/ --observation_time=7 --least_num=10 --up_num=100
python -u gen_walks.py --rawdataset=dataset_dblp.txt --dataset=dblp3/ --observation_time=7 --least_num=10 --up_num=100
python -u gen_run.py --rawdataset=dataset_dblp.txt --dataset=dblp3/ --observation_time=7 --least_num=10 --up_num=100
python -u run.py --rawdataset=dataset_dblp.txt --dataset=dblp3/ --observation_time=7 --least_num=10 --up_num=100 > 7.log 


