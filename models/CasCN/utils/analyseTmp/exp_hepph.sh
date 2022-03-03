python -u gen_cas.py --rawdataset=hepph.txt --dataset=hepph1/ --observation_time=1080 --interval=90 --up_num=100 --least_num=5 
python -u gen_graph_signal.py --rawdataset=hepph.txt --dataset=hepph1/ --observation_time=1080 --interval=90 --up_num=100 --least_num=5 
python -u run.py --rawdataset=hepph.txt --dataset=hepph1/ --observation_time=1080 --interval=90 --up_num=100 --least_num=5 > 1080.log

python -u gen_cas.py --rawdataset=hepph.txt --dataset=hepph2/ --observation_time=1800 --interval=180 --up_num=100  --least_num=5
python -u gen_graph_signal.py --rawdataset=hepph.txt --dataset=hepph2/ --observation_time=1800 --interval=180 --up_num=100  --least_num=5
python -u run.py --rawdataset=hepph.txt --dataset=hepph2/ --observation_time=1800 --interval=180 --up_num=100  --least_num=5 > 1800.log

python -u gen_cas.py --rawdataset=hepph.txt --dataset=hepph3/ --observation_time=2520 --interval=180 --up_num=100 --least_num=5
python -u gen_graph_signal.py --rawdataset=hepph.txt --dataset=hepph3/ --observation_time=2520 --interval=180 --up_num=100 --least_num=5
python -u run.py --rawdataset=hepph.txt --dataset=hepph3/ --observation_time=2520 --interval=180 --up_num=100 --least_num=5 > 2520.log