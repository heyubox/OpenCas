python -u  gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation0_long/ --observation_time=1080  --up_num=1000 --least_num=10
python -u gen_run.py --rawdataset=dataset_citation.txt --dataset=citation0_long/ --observation_time=1080  --up_num=1000 --least_num=10
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation0_long/ --observation_time=1080  --up_num=1000 --least_num=10 > 1080_long.log  

python -u  gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation1_long/ --observation_time=1800  --up_num=1000 --least_num=10
python -u gen_run.py --rawdataset=dataset_citation.txt --dataset=citation1_long/ --observation_time=1800  --up_num=1000 --least_num=10
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation1_long/ --observation_time=1800  --up_num=1000 --least_num=10 > 1800_long.log 

python -u  gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation2_long/ --observation_time=2520  --up_num=1000 --least_num=10
python -u gen_run.py --rawdataset=dataset_citation.txt --dataset=citation2_long/ --observation_time=2520  --up_num=1000 --least_num=10
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation2_long/ --observation_time=2520  --up_num=1000 --least_num=10 > 2520_long.log 

# # python -u  gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation3_long/ --observation_time=3240  --up_num=1000 --least_num=10
# # python -u gen_run.py --rawdataset=dataset_citation.txt --dataset=citation3_long/ --observation_time=3240  --up_num=1000 --least_num=10
# python -u run.py --rawdataset=dataset_citation.txt --dataset=citation3_long/ --observation_time=3240  --up_num=1000 --least_num=10 > 3240_long.log 

# # python -u  gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation4_long/ --observation_time=3960  --up_num=1000 --least_num=10
# # python -u gen_run.py --rawdataset=dataset_citation.txt --dataset=citation4_long/ --observation_time=3960  --up_num=1000 --least_num=10
# python -u run.py --rawdataset=dataset_citation.txt --dataset=citation4_long/ --observation_time=3960  --up_num=1000  --least_num=10 > 3960_long.log 