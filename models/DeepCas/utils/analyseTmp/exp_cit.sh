python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation0/ --observation_time=1080 --up_num=100 --least_num=10
python -u gen_walks.py --rawdataset=dataset_citation.txt --dataset=citation0/ --observation_time=1080 --up_num=100 --least_num=10
python -u gen_run.py --rawdataset=dataset_citation.txt --dataset=citation0/ --observation_time=1080 --up_num=100 --least_num=10
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation0/ --observation_time=1080 --up_num=100  --least_num=10> 1080.log

python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation1/ --observation_time=1800 --up_num=100 --least_num=10
python -u gen_walks.py --rawdataset=dataset_citation.txt --dataset=citation1/ --observation_time=1800 --up_num=100 --least_num=10
python -u gen_run.py --rawdataset=dataset_citation.txt --dataset=citation1/ --observation_time=1800 --up_num=100 --least_num=10
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation1/ --observation_time=1800 --up_num=100  --least_num=10 > 1800.log

python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation2/ --observation_time=2520 --up_num=100 --least_num=10
python -u gen_walks.py --rawdataset=dataset_citation.txt --dataset=citation2/ --observation_time=2520 --up_num=100 --least_num=10
python -u gen_run.py --rawdataset=dataset_citation.txt --dataset=citation2/ --observation_time=2520 --up_num=100 --least_num=10
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation2/ --observation_time=2520 --up_num=100  --least_num=10> 2520.log 

python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation3/ --observation_time=3240 --up_num=100 --least_num=10
python -u gen_walks.py --rawdataset=dataset_citation.txt --dataset=citation3/ --observation_time=3240 --up_num=100 --least_num=10
python -u gen_run.py --rawdataset=dataset_citation.txt --dataset=citation3/ --observation_time=3240 --up_num=100 --least_num=10
python -u run.py --rawdataset=dataset_citation.txt --dataset=citation3/ --observation_time=3240 --up_num=100  --least_num=10 > 3240.log 

# python -u gen_cas.py --rawdataset=dataset_citation.txt --dataset=citation4/ --observation_time=3960  --up_num=100
# python -u gen_walks.py --rawdataset=dataset_citation.txt --dataset=citation4/ --observation_time=3960  --up_num=100
# python -u gen_run.py --rawdataset=dataset_citation.txt --dataset=citation4/ --observation_time=3960  --up_num=100
# python -u run.py --rawdataset=dataset_citation.txt --dataset=citation4/ --observation_time=3960  --up_num=100 > 3960.log 