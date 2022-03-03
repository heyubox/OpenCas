# python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation0/ --observation_time=1080 --least_num=10 --up_num=100
# python -u gen_emb.py --rawdataset=dataset_citation.txt --dataset=citation0/ --observation_time=1080 --least_num=10 --up_num=100
# python -u run.py --rawdataset=dataset_citation.txt --dataset=citation0/ --observation_time=1080 --least_num=10 --up_num=100 > 1080.log 

# python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation1/ --observation_time=1800 --least_num=10 --up_num=100
# python -u gen_emb.py --rawdataset=dataset_citation.txt --dataset=citation1/ --observation_time=1800 --least_num=10 --up_num=100
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation1/ --observation_time=1800 --least_num=10 --up_num=100 > 1800.log 

# python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation2/ --observation_time=2520 --least_num=10 --up_num=100
# python -u gen_emb.py --rawdataset=dataset_citation.txt --dataset=citation2/ --observation_time=2520 --least_num=10 --up_num=100
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation2/ --observation_time=2520 --least_num=10 --up_num=100 > 2520.log 

# python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation3/ --observation_time=3240 --least_num=10 --up_num=100
# python -u gen_emb.py --rawdataset=dataset_citation.txt --dataset=citation3/ --observation_time=3240 --least_num=10 --up_num=100
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation3/ --observation_time=3240 --least_num=10 --up_num=100 > 3240.log 

# python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation4/ --observation_time=3960 --least_num=10 --up_num=100
# python -u gen_emb.py --rawdataset=dataset_citation.txt --dataset=citation4/ --observation_time=3960 --least_num=10 --up_num=100
# python -u run.py --rawdataset=dataset_citation.txt --dataset=citation4/ --observation_time=3960 --least_num=10 --up_num=100 > 3960.log 