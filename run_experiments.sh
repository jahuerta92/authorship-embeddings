python3 run_experiment.py --devices 1 --model lstm --mails --scheduler enable
python3 run_experiment.py --devices 1 --model max --mails --scheduler enable
python3 run_experiment.py --devices 1 --model mean --mails --scheduler enable

python3 run_experiment.py --devices 1 --model lstm --books --batch_size 1024 --vbatch_size 100 --scheduler enable
python3 run_experiment.py --devices 1 --model max --books --batch_size 1024 --vbatch_size 100 --scheduler enable
python3 run_experiment.py --devices 1 --model mean --books --batch_size 1024 --vbatch_size 100 --scheduler enable

python3 run_experiment.py --devices 1 --model lstm --blogs --batch_size 2048 --vbatch_size 100 --scheduler enable
python3 run_experiment.py --devices 1 --model max --blogs --batch_size 2048 --vbatch_size 100 --scheduler enable
python3 run_experiment.py --devices 1 --model mean --blogs --batch_size 2048 --vbatch_size 100 --scheduler enable

python3 run_experiment.py --devices 1 --model lstm --books --mails --blogs --batch_size 2048 --vbatch_size 1000 --scheduler enable
python3 run_experiment.py --devices 1 --model max --books --mails --blogs --batch_size 2048 --vbatch_size 1000 --scheduler enable
python3 run_experiment.py --devices 1 --model mean --books --mails --blogs --batch_size 2048 --vbatch_size 1000 --scheduler enable

# python3 run_experiment.py --devices 1 --model experimental --books --mails --blogs --scheduler enable --batch_size 32

