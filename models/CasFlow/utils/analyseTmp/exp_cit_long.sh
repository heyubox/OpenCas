# python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation0_long/ --observation_time=1080 --least_num=5 --up_num=1000 --random_seed=0
# python -u gen_emb.py --rawdataset=dataset_citation.txt --dataset=citation0_long/ --observation_time=1080 --least_num=5 --up_num=1000 --random_seed=0
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation0_long/ --observation_time=1080 --least_num=5 --up_num=1000 > 1080_long.log 

# python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation1_long/ --observation_time=1800 --least_num=5 --up_num=1000 --random_seed=0
# python -u gen_emb.py --rawdataset=dataset_citation.txt --dataset=citation1_long/ --observation_time=1800 --least_num=5 --up_num=1000 --random_seed=0
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation1_long/ --observation_time=1800 --least_num=5 --up_num=1000 > 1800_long.log 

# python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation2_long/ --observation_time=2520 --least_num=5 --up_num=1000 --random_seed=0
# python -u gen_emb.py --rawdataset=dataset_citation.txt --dataset=citation2_long/ --observation_time=2520 --least_num=5 --up_num=1000 --random_seed=0
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation2_long/ --observation_time=2520 --least_num=5 --up_num=1000 > 2520_long.log 

# python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation3_long/ --observation_time=3240 --least_num=5 --up_num=1000 --random_seed=0
# python -u gen_emb.py --rawdataset=dataset_citation.txt --dataset=citation3_long/ --observation_time=3240 --least_num=5 --up_num=1000 --random_seed=0
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation3_long/ --observation_time=3240 --least_num=5 --up_num=1000 > 3240_long.log 

# python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation4_long/ --observation_time=3960 --least_num=5 --up_num=1000 --random_seed=0
# python -u gen_emb.py --rawdataset=dataset_citation.txt --dataset=citation4_long/ --observation_time=3960 --least_num=5 --up_num=1000 --random_seed=0
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation4_long/ --observation_time=3960 --least_num=5 --up_num=1000 > 3960_long.log 