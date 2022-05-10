###############################################################################
# Imports #####################################################################
###############################################################################
import pandas as pd
import wandb

from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from ast import literal_eval
from tqdm import tqdm

from data import build_dataset
from model import (ContrastiveMaxDenseHead,
                   ContrastiveMeanDenseHead, 
                   ContrastiveLSTMHead,
                   )

###############################################################################
# Runtime parameters ##########################################################
###############################################################################
arg_parser = ArgumentParser(description='Run an experiment.')
arg_parser.add_argument('--books', action='store_true', help='Use books dataset')
arg_parser.add_argument('--mails', action='store_true', help='Use mails dataset')
arg_parser.add_argument('--blogs', action='store_true', help='Use blogs dataset')
arg_parser.add_argument('--model', type=str, required=True, help='Model type',
                        choices=['max', 'mean', 'lstm'],
                        )
arg_parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
arg_parser.add_argument('--devices', type=int, nargs='+', default=[0], help='Devices to use')

args = arg_parser.parse_args()

BATCH_SIZE = args.batch_size
DEVICES = args.devices
MODEL_TYPE = args.model
if MODEL_TYPE == 'max':
    MODEL = ContrastiveMaxDenseHead
elif MODEL_TYPE == 'mean':
    MODEL = ContrastiveMeanDenseHead
elif MODEL_TYPE == 'lstm':
    MODEL = ContrastiveLSTMHead

TRAIN_FILES = {'books': 'local_data/book_train.csv',
               'mails': 'local_data/mail_train.csv',
               'blogs': 'local_data/blog_train.csv',
                }
TEST_FILES = {'books': 'local_data/book_test.csv',
              'mails': 'local_data/mail_test.csv',
              'blogs': 'local_data/blog_test.csv',
                }
USED_FILES = []
if args.books:
    USED_FILES.append('books')
if args.mails:
    USED_FILES.append('mails')
if args.blogs:
    USED_FILES.append('blogs')

VALID_BATCH_SIZE = 100
VALID_STEPS = 5000
CHUNK_SIZE = 512

LEARNING_RATE = 5e-5
WEIGHT_DECAY = .01
TRAINING_STEPS = 10000
WARMPU_STEPS = int(TRAINING_STEPS*.1)

###############################################################################
# Main method #################################################################
###############################################################################

def main():
    # Load preferred datasets
    train_datasets, test_datasets = [], []
    tqdm.pandas()
    for file_code in USED_FILES:
        print(f'Loading {file_code} dataset...')
        train_file = pd.read_csv(TRAIN_FILES[file_code])
        test_file = pd.read_csv(TEST_FILES[file_code])

        train_file['unique_id'] = train_file.index.astype(str) + f'_{file_code}'
        test_file['unique_id'] = test_file.index.astype(str) + f'_{file_code}'

        train_file.pretokenized_text = train_file.pretokenized_text.progress_apply(literal_eval)
        test_file.pretokenized_text = test_file.pretokenized_text.progress_apply(literal_eval)

        train_datasets.append(train_file[['unique_id', 'id', 'pretokenized_text', 'decoded_text']].sample(frac=1))
        test_datasets.append(test_file[['unique_id', 'id', 'pretokenized_text', 'decoded_text']].sample(frac=1))
    
    train = pd.concat(train_datasets)
    test = pd.concat(test_datasets)

    del train_datasets
    del test_datasets

    # Build dataset
    train_data = build_dataset(train,
                               steps=len(train),
                               batch_size=BATCH_SIZE, 
                               max_len=CHUNK_SIZE)
    test_data = build_dataset(test, 
                              steps=VALID_STEPS, 
                              batch_size=VALID_BATCH_SIZE, 
                              num_workers=2, 
                              prefetch_factor=2, 
                              max_len=CHUNK_SIZE)

    # Name model
    model_datasets = '+'.join(USED_FILES)
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_name = f'{date_time}_{MODEL_TYPE}_{model_datasets}'
    print(f'Saving model to {save_name}')

    # Callbacks
    wandb.login()
    wandb_logger = WandbLogger(name=save_name, project="author_profiling_extension")
    checkpoint_callback = ModelCheckpoint('model',
                                          filename=save_name,
                                          monitor=None,
                                          every_n_val_epochs=1,
                                          )
    lr_monitor = LearningRateMonitor('step')

    # Define training arguments
    trainer = Trainer(devices=DEVICES,
                    max_steps=TRAINING_STEPS,
                    accelerator='gpu',
                    log_every_n_steps=1,
                    flush_logs_every_n_steps=500,
                    logger=wandb_logger,
                    strategy='dp',
                    precision=16,
                    val_check_interval=500,
                    callbacks=[checkpoint_callback, lr_monitor],
                    )

    # Define model
    base_transformer = AutoModel.from_pretrained('roberta-large')
    train_model = MODEL(base_transformer,
                        learning_rate=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        num_warmup_steps=WARMPU_STEPS,
                        num_training_steps=TRAINING_STEPS)

    # Fit and log
    trainer.fit(train_model, train_data, test_data)
    wandb.finish()

if __name__ == '__main__':
    main()