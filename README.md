# doduo-submission

This anonymous repository contains the codebase of our SIGMOD submission ``Annotating Columns with Pre-trained Language Models'''

## Quick Demo

Please check out our [Google Colab Demo](https://colab.research.google.com/drive/1uXI5d2evZ30Tm8PV3ve1YDvFjVXZHV4z?usp=sharing) to test out a trained Doduo model with custom tables.  


## Installation 

```console
$ git clone
$ cd doduo-dev
$ pip install -r requirements.txt 
```

With `conda`, create a virtual environment and install required packages as below:

```console
$ conda create --name doduo python=3.7.10
$ conda activate doduo
$ pip install -r requirements.txt
```


## Data Preparation

Run `download.sh` to download processed data for the VizNet corpus. It will also create `data` directory.
Please note that by downloading these data, you are accepting [original license requirements](https://github.com/mitmedialab/viznet). 


```console
$ bash download.sh
```

You will see the following files in `data` 

```console
$ ls data
msato_cv_0.csv
msato_cv_1.csv
msato_cv_2.csv
msato_cv_3.csv
msato_cv_4.csv
sato_cv_0.csv
sato_cv_1.csv
sato_cv_2.csv
sato_cv_3.csv
sato_cv_4.csv
table_col_type_serialized.pkl
table_rel_extraction_serialized.pkl
```


For the WikiTable corpus, download the following files from [here](https://github.com/sunlab-osu/TURL#data) and save under `data/turl_dataset`. 

```console
$ tree data/turl_dataset
data/turl_dataset
├── dev.table_col_type.json
├── dev.table_rel_extraction.json
├── test.table_col_type.json
├── test.table_rel_extraction.json
├── train.table_col_type.json
└── train.table_rel_extraction.json
```


## Usage

- `doduo/train_multi.py`
```console
usage: train_multi.py [-h] [--shortcut_name SHORTCUT_NAME]
                      [--max_length MAX_LENGTH] [--batch_size BATCH_SIZE]
                      [--epoch EPOCH] [--random_seed RANDOM_SEED]
                      [--num_classes NUM_CLASSES] [--multi_gpu] [--fp16]
                      [--warmup WARMUP] [--lr LR]
                      [--tasks {sato0,sato1,sato2,sato3,sato4,msato0,msato1,msato2,msato3,msato4,turl,turl-re} [{sato0,sato1,sato2,sato3,sato4,msato0,msato1,msato2,msato3,msato4,turl,turl-re} ...]]
                      [--colpair]
                      [--train_ratios TRAIN_RATIOS [TRAIN_RATIOS ...]]
                      [--from_scratch] [--single_col]

optional arguments:
  -h, --help            show this help message and exit
  --shortcut_name SHORTCUT_NAME
                        Huggingface model shortcut name
  --max_length MAX_LENGTH
                        The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
  --batch_size BATCH_SIZE
                        Batch size
  --epoch EPOCH         Number of epochs for training
  --random_seed RANDOM_SEED
                        Random seed
  --num_classes NUM_CLASSES
                        Number of classes
  --multi_gpu           Use multiple GPU
  --fp16                Use FP16
  --warmup WARMUP       Warmup ratio
  --lr LR               Learning rate
  --tasks {sato0,sato1,sato2,sato3,sato4,msato0,msato1,msato2,msato3,msato4,turl,turl-re,turl-sch,turl-re-sch} [{sato0,sato1,sato2,sato3,sato4,msato0,msato1,msato2,msato3,msato4,turl,turl-re,turl-sch,turl-re-sch} ...]
                        Task name {sato, turl, turl-re, turl-sch, turl-re-sch}
  --colpair             Use column pair embedding
  --train_ratios TRAIN_RATIOS [TRAIN_RATIOS ...]
                        e.g., --train_ratios turl=0.8 turl-re=0.1
  --from_scratch        Training from scratch
  --single_col          Training with single column model
```
- `doduo/predict_multi.py`

```console
$ predict_multi.py <MODEL_PATH>
```

## Running Experiments

```console
$ python doduo/train_multi.py --tasks turl --max_length 32 --batch_size 16 
```

To specify GPU, use `CUDA_VISIBLE_DEVICES` environment variable. For example,

```console
$ CUDA_VISIBLE_DEVICES=0 python doduo/train_multi.py --tasks turl --max_length 32 --batch_size 16 
```

After training, you will see the following files in the `./model` directory. 

```console
$ ls model
turlturl-re-colpair_mosato_bert_bert-base-uncased-bs16-ml-32__turl-1.00_turl-re-1.00=turl-re_best_micro_f1.pt
turlturl-re-colpair_mosato_bert_bert-base-uncased-bs16-ml-32__turl-1.00_turl-re-1.00=turl_best_micro_f1.pt
```

You can run the prediction script as below  

```console
$ python doduo/predict_multi.py turlturl-re-colpair_mosato_bert_bert-base-uncased-bs16-ml-32__turl-1.00_turl-re-1.00=turl
```

This creates a JSON file that contains evaluation metrics. 

```console
$ ls eval 
turlturl-re-colpair_mosato_bert_bert-base-uncased-bs16-ml-32__turl-1.00_turl-re-1.00=turl.json
```
