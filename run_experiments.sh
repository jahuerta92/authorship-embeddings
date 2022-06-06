python3 run_experiment.py --devices 0 --model lstm --books --mails --blogs --batch_size 2048 --vbatch_size 1000
python3 run_experiment.py --devices 0 --model max --books --mails --blogs --batch_size 2048 --vbatch_size 1000
python3 run_experiment.py --devices 0 --model mean --books --mails --blogs --batch_size 2048 --vbatch_size 1000

python3 run_experiment.py --devices 0 --model lstm --books --batch_size 1024 --vbatch_size 100
python3 run_experiment.py --devices 0 --model max --books --batch_size 1024 --vbatch_size 100
python3 run_experiment.py --devices 0 --model mean --books --batch_size 1024 --vbatch_size 100

python3 run_experiment.py --devices 0 --model lstm --mails
python3 run_experiment.py --devices 0 --model max --mails
python3 run_experiment.py --devices 0 --model mean --mails 

python3 run_experiment.py --devices 0 --model lstm --blogs --batch_size 2048 --vbatch_size 100
python3 run_experiment.py --devices 0 --model max --blogs --batch_size 2048 --vbatch_size 100
python3 run_experiment.py --devices 0 --model mean --blogs --batch_size 2048 --vbatch_size 100

# python3 run_experiment.py --devices 1 --model experimental --books --mails --blogs --scheduler enable --batch_size 32

