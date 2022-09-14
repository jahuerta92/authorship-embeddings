import pandas as pd
from data import build_dataset, build_supervised_dataset
from transformers import AutoTokenizer
import wandb

from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning import Trainer

from model_experimental import SupervisedContrastiveTransformerLast

                                
USED_FILES = ['local_data/reddit_train.csv']#, 'local_data/book_train.csv', 'local_data/blog_train.csv']
    
train_datasets = []
for file in USED_FILES:
    train_file = pd.read_csv(file)
    train_file['unique_id'] = train_file.index.astype(str) + f'{file}'
    train_datasets.append(train_file[['unique_id', 'id', 'decoded_text']])
    
train = pd.concat(train_datasets).sample(frac=1.)
test = pd.read_csv('local_data/reddit_test.csv')

train.columns = ['unique_id', 'id', 'decoded_text']
test.columns = ['id', 'decoded_text', 'subreddit']

train = train.drop_duplicates(subset=["decoded_text"], keep=False)

test['unique_id'] = test.index.astype(str)


BATCH_SIZE = 32
VIEW_SIZE = 16
MINIBATCH_SIZE = 128
VALID_BATCH_SIZE = 16
CHUNK_SIZE = 512
TRAINING_STEPS = 1000
VALIDATION_STEPS = 500
WARMUP_STEPS = 0

train_data = build_supervised_dataset(train,
                           steps=TRAINING_STEPS*BATCH_SIZE,
                           batch_size=BATCH_SIZE,
                           num_workers=4, 
                           prefetch_factor=4,
                           max_len=CHUNK_SIZE,
                           views=VIEW_SIZE,
                           tokenizer = AutoTokenizer.from_pretrained('roberta-base'),
                           )
test_data = build_supervised_dataset(test, 
                          steps=VALIDATION_STEPS*VALID_BATCH_SIZE, 
                          batch_size=VALID_BATCH_SIZE, 
                          num_workers=4, 
                          prefetch_factor=4, 
                          max_len=CHUNK_SIZE,
                          views=VIEW_SIZE,
                          shuffle=False,
                          tokenizer = AutoTokenizer.from_pretrained('roberta-base'),
                          )

# Name model
date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_name = f'{date_time}_reddit_supconloss'
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
train_model = SupervisedContrastiveTransformerLast(base_transformer,
                                                   learning_rate=2e-3,
                                                   weight_decay=0.0,
                                                   num_warmup_steps=TRAINING_STEPS*.0,
                                                   num_training_steps=TRAINING_STEPS,
                                                   enable_scheduler=True,
                                                   minibatch_size=MINIBATCH_SIZE,
                                                   unfreeze=6,
                                                   )

trainer.fit(train_model, train_data, test_data)#, ckpt_path='model/2022-06-22_15-28-24_reddit-v1.ckpt')
wandb.finish()
