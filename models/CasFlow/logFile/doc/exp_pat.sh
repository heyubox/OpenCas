python -u gen_cas.py --rawdataset=dataset_patent.txt --dataset=patent0/ --observation_time=1080 --least_num=5 --up_num=100
python -u gen_emb.py --rawdataset=dataset_patent.txt --dataset=patent0/ --observation_time=1080 --least_num=5 --up_num=100
python -u run.py --rawdataset=dataset_patent.txt --dataset=patent0/ --observation_time=1080 --least_num=5 --up_num=100 > 1080.log 

python -u gen_cas.py --rawdataset=dataset_patent.txt --dataset=patent1/ --observation_time=1800 --least_num=5 --up_num=100
python -u gen_emb.py --rawdataset=dataset_patent.txt --dataset=patent1/ --observation_time=1800 --least_num=5 --up_num=100
python -u run.py --rawdataset=dataset_patent.txt --dataset=patent1/ --observation_time=1800 --least_num=5 --up_num=100 > 1800.log 

python -u gen_cas.py --rawdataset=dataset_patent.txt --dataset=patent2/ --observation_time=2520 --least_num=5 --up_num=100
python -u gen_emb.py --rawdataset=dataset_patent.txt --dataset=patent2/ --observation_time=2520 --least_num=5 --up_num=100
python -u run.py --rawdataset=dataset_patent.txt --dataset=patent2/ --observation_time=2520 --least_num=5 --up_num=100 > 2520.log 

python -u gen_cas.py --rawdataset=dataset_patent.txt --dataset=patent3/ --observation_time=3240 --least_num=5 --up_num=100
python -u gen_emb.py --rawdataset=dataset_patent.txt --dataset=patent3/ --observation_time=3240 --least_num=5 --up_num=100
python -u run.py --rawdataset=dataset_patent.txt --dataset=patent3/ --observation_time=3240 --least_num=5 --up_num=100 > 3240.log 

python -u gen_cas.py --rawdataset=dataset_patent.txt --dataset=patent4/ --observation_time=3960 --least_num=5 --up_num=100
python -u gen_emb.py --rawdataset=dataset_patent.txt --dataset=patent4/ --observation_time=3960 --least_num=5 --up_num=100
python -u run.py --rawdataset=dataset_patent.txt --dataset=patent4/ --observation_time=3960 --least_num=5 --up_num=100 > 3960.log 