###############################################################################
# Imports #####################################################################
###############################################################################
import pandas as pd
import numpy as np
import wandb

from datetime import datetime
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
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
from model_experimental import (ContrastiveTransformer,
                                ContrastiveLSTMTransformer,
                                ContrastiveMeanDenseTransformer,
                                ContrastiveMaxDenseTransformer,
                                )

###############################################################################
# Runtime parameters ##########################################################
###############################################################################
arg_parser = ArgumentParser(description='Run an experiment.')
arg_parser.add_argument('--books', action='store_true', help='Use books dataset')
arg_parser.add_argument('--mails', action='store_true', help='Use mails dataset')
arg_parser.add_argument('--blogs', action='store_true', help='Use blogs dataset')
arg_parser.add_argument('--model', type=str, required=True, help='Model type',
                        choices=['max', 'mean', 'lstm', 'experimental', 'experimental_lstm'],
                        )
arg_parser.add_argument('--scheduler', type=str, default='none', help='Model type',
                        choices=['enable', 'none'],
                        )
arg_parser.add_argument('--transformer', type=str, default='roberta-large', help='Model type',
                        choices=['roberta-large', 'roberta-base', 'distilroberta-base', 'google/t5-v1_1-base'],
                        )
arg_parser.add_argument('--batch_size', type=int, default=0, help='Batch size')
arg_parser.add_argument('--vbatch_size', type=int, default=0, help='Validation batch size')
arg_parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
arg_parser.add_argument('--devices', type=int, nargs='+', default=[0], help='Devices to use')

args = arg_parser.parse_args()

BATCH_SIZE = args.batch_size
VALID_BATCH_SIZE = args.vbatch_size
ENABLE_SCHEDULER = args.scheduler == 'enable'
DEVICES = args.devices
MODEL_TYPE = args.model
BASE_CODE = args.transformer
if MODEL_TYPE == 'max':
    MODEL = ContrastiveMaxDenseTransformer
elif MODEL_TYPE == 'mean':
    MODEL = ContrastiveMeanDenseTransformer
elif MODEL_TYPE == 'lstm':
    MODEL = ContrastiveLSTMHead
elif MODEL_TYPE == 'experimental':
    MODEL = ContrastiveTransformer
elif MODEL_TYPE == 'experimental_lstm':
    MODEL = ContrastiveLSTMTransformer

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

MINIBATCH_SIZE = 512
VALID_STEPS = 50
CHUNK_SIZE = 512
LEARNING_RATE = 5e-3
DROPOUT = .1
WEIGHT_DECAY = .01
LABEL_SMOOTHING = .0
TRAINING_STEPS = 3000
WARMUP_STEPS = 0 #1000 #int(TRAINING_STEPS*.1)

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

        train_datasets.append(train_file[['unique_id', 'id', 'pretokenized_text', 'decoded_text']])
        test_datasets.append(test_file[['unique_id', 'id', 'pretokenized_text', 'decoded_text']])
    
    train = pd.concat(train_datasets).sample(frac=1.)
    test = pd.concat(test_datasets)

    del train_datasets
    del test_datasets

    # Build dataset
    n_auth = len(train.id.unique()) if BATCH_SIZE == 0 else BATCH_SIZE
    n_auth_v = len(test.id.unique()) if VALID_BATCH_SIZE == 0 else VALID_BATCH_SIZE

    # get closest power of 2 to n_auth
    n_auth = int(2 ** np.floor(np.log(n_auth)/np.log(2)))
    n_auth_v = int(2 ** np.floor(np.log(n_auth_v)/np.log(2)))

    print(f'Batch size equals: {n_auth}')
    train_data = build_dataset(train,
                               steps=TRAINING_STEPS*n_auth,
                               batch_size=n_auth,
                               num_workers=8, 
                               prefetch_factor=8,
                               max_len=CHUNK_SIZE,
                               tokenizer = AutoTokenizer.from_pretrained(BASE_CODE),
                               mode='text')
    test_data = build_dataset(test, 
                              steps=VALID_STEPS*n_auth_v, 
                              batch_size=n_auth_v, 
                              num_workers=8, 
                              prefetch_factor=8, 
                              max_len=CHUNK_SIZE,
                              tokenizer = AutoTokenizer.from_pretrained(BASE_CODE),
                              mode='text')

    # Name model
    model_datasets = '+'.join(USED_FILES)
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_name = f'final_{date_time}_{MODEL_TYPE}_{model_datasets}'
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
                    #strategy='dp',
                    precision=16,
                    val_check_interval=100,
                    callbacks=[checkpoint_callback, lr_monitor],
                    )

    # Define model
    if ('T0' in BASE_CODE) or ('t5-v1_1' in BASE_CODE):
        base_transformer = T5EncoderModel.from_pretrained(BASE_CODE)

    else:
        base_transformer = AutoModel.from_pretrained(BASE_CODE, 
                                                     hidden_dropout_prob = DROPOUT, 
                                                     attention_probs_dropout_prob = DROPOUT)
    train_model = MODEL(base_transformer,
                        learning_rate=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        num_warmup_steps=WARMUP_STEPS,
                        num_training_steps=TRAINING_STEPS,
                        enable_scheduler=ENABLE_SCHEDULER,
                        minibatch_size=MINIBATCH_SIZE,)

    # Fit and log
    trainer.fit(train_model, train_data, test_data)
    wandb.finish()

if __name__ == '__main__':
    main()