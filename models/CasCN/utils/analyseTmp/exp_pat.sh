python -u gen_cas.py --rawdataset=dataset_patent.txt --dataset=patent0/ --observation_time=1080 --interval=120 --up_num=100 --least_num=5 
python -u gen_graph_signal.py --rawdataset=dataset_patent.txt --dataset=patent0/ --observation_time=1080 --interval=120 --up_num=100 --least_num=5 
python -u run.py --rawdataset=dataset_patent.txt --dataset=patent0/ --observation_time=1080 --interval=120 --up_num=100 --least_num=5  > 1080.log

python -u gen_cas.py --rawdataset=dataset_patent.txt --dataset=patent1/ --observation_time=1800 --interval=180 --up_num=100 --least_num=5 
python -u gen_graph_signal.py --rawdataset=dataset_patent.txt --dataset=patent1/ --observation_time=1800 --interval=180 --up_num=100 --least_num=5 
python -u run.py --rawdataset=dataset_patent.txt --dataset=patent1/ --observation_time=1800 --interval=180 --up_num=100 --least_num=5  > 1800.log

python -u gen_cas.py --rawdataset=dataset_patent.txt --dataset=patent2/ --observation_time=2520 --interval=180 --up_num=100  --least_num=5
python -u gen_graph_signal.py --rawdataset=dataset_patent.txt --dataset=patent2/ --observation_time=2520 --interval=180 --up_num=100  --least_num=5
python -u run.py --rawdataset=dataset_patent.txt --dataset=patent2/ --observation_time=2520 --interval=180 --up_num=100  --least_num=5 > 2520.log

python -u gen_cas.py --rawdataset=dataset_patent.txt --dataset=patent3/ --observation_time=3240 --interval=360 --up_num=100 --least_num=5 
python -u gen_graph_signal.py --rawdataset=dataset_patent.txt --dataset=patent3/ --observation_time=3240 --interval=360 --up_num=100  --least_num=5
python -u run.py --rawdataset=dataset_patent.txt --dataset=patent3/ --observation_time=3240 --interval=360 --up_num=100  --least_num=5 > 3240.log

python -u gen_cas.py --rawdataset=dataset_patent.txt --dataset=patent4/ --observation_time=3960 --interval=360 --up_num=100 --least_num=5 
python -u gen_graph_signal.py --rawdataset=dataset_patent.txt --dataset=patent4/ --observation_time=3960 --interval=360 --up_num=100  --least_num=5
python -u run.py --rawdataset=dataset_patent.txt --dataset=patent3/ --observation_time=3960 --interval=360 --up_num=100  --least_num=5 > 3960.log