import pandas as pd
from data import build_dataset
from transformers import AutoTokenizer
import wandb

from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer

from model_experimental import (ContrastiveLSTMTransformer,
                                )

train = pd.read_csv('local_data/reddit_train.csv').sample(frac=1.)
test = pd.read_csv('local_data/reddit_test.csv')

train.columns = ['id', 'decoded_text', 'subreddit']
test.columns = ['id', 'decoded_text', 'subreddit']

train['unique_id'] = train.index.astype(str)
test['unique_id'] = test.index.astype(str)


BATCH_SIZE = 4096
VALID_BATCH_SIZE = 1000
CHUNK_SIZE = 512
TRAINING_STEPS = 3000
VALIDATION_STEPS = 500
WARMUP_STEPS = 0

train_data = build_dataset(train,
                           steps=TRAINING_STEPS*BATCH_SIZE,
                           batch_size=BATCH_SIZE,
                           num_workers=8, 
                           prefetch_factor=8,
                           max_len=CHUNK_SIZE,
                           tokenizer = AutoTokenizer.from_pretrained('roberta-base'),
                           mode='text')
test_data = build_dataset(test, 
                          steps=VALIDATION_STEPS*VALID_BATCH_SIZE, 
                          batch_size=VALID_BATCH_SIZE, 
                          num_workers=4, 
                          prefetch_factor=4, 
                          max_len=CHUNK_SIZE,
                          tokenizer = AutoTokenizer.from_pretrained('roberta-base'),
                          mode='text')


# Name model
date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_name = f'{date_time}_reddit'
print(f'Saving model to {save_name}')

wandb.login()
wandb_logger = WandbLogger(name=save_name, project="author_profiling_reddit")
checkpoint_callback = ModelCheckpoint('model',
                                      filename=save_name,
                                      monitor=None,
                                      every_n_val_epochs=1,
                                      )
lr_monitor = LearningRateMonitor('step')

# Define training arguments
trainer = Trainer(devices=[0],
                  max_steps=TRAINING_STEPS,
                  accelerator='gpu',
                  log_every_n_steps=1,
                  flush_logs_every_n_steps=500,
                  logger=wandb_logger,
                  precision=16,
                  val_check_interval=250,
                  callbacks=[checkpoint_callback, lr_monitor],
                  )

# Define model
base_transformer = AutoModel.from_pretrained('roberta-large')
train_model = ContrastiveLSTMTransformer(base_transformer,
                                         learning_rate=1e-2,
                                         weight_decay=.01,
                                         num_warmup_steps=0,
                                         num_training_steps=TRAINING_STEPS,
                                         enable_scheduler=True,
                                         minibatch_size=256,)

trainer.fit(train_model, train_data, test_data)
wandb.finish()