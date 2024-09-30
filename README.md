# JIT-Contrast-replication-package



This repository contains source code that we used to perform experiment in paper titled "JIT-Contrast: Enhancing Just-In-Time Defect Prediction with Contrastive Learning and Feature Fusion".


Please follow the steps below to reproduce the result


## Environment Setup

### Python Environment Setup

Run the following command in terminal (or command line) to prepare virtual environment

```shell
conda env create --file requirements.yml
conda activate jitcontrast
```

### R Environment Setup

Download the following package: `tidyverse`, `gridExtra`, `ModelMetrics`, `caret`, `reshape2`, `pROC`, `effsize`, `ScottKnottESD`

## Experiment Result Replication Guide

Before the replication, you need to unzip `data.zip` file firstly.



### **JIT-Contrast Implementation**

To train JIT-Contrast, run the following command:

```shell
python -m JITContrast.concatCL.run \
    --output_dir=model/jitcl/saved_models_concat_cl/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_train \
    --train_data_file data/jitcl/changes_train.pkl data/jitcl/features_train.pkl \
    --eval_data_file data/jitcl/changes_valid.pkl data/jitcl/features_valid.pkl\
    --test_data_file data/jitcl/changes_test.pkl data/jitcl/features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 24 \
    --eval_batch_size 128 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --feature_size 14 \
    --patience 10 \
    --seed 42 2>&1| tee model/jitcl/saved_models_concat_cl/train.log

```

To obtain the evaluation, run the following command:

```shell
python -m JITFine.concatCL.run \
    --output_dir=model/jitcl/saved_models_concat_cl/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_test \
    --train_data_file data/jitcl/changes_train.pkl data/jitcl/features_train.pkl \
    --eval_data_file data/jitcl/changes_valid.pkl data/jitcl/features_valid.pkl\
    --test_data_file data/jitcl/changes_test.pkl data/jitcl/features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 256 \
    --eval_batch_size 25 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --only_adds \
    --buggy_line_filepath=data/jitcl/changes_complete_buggy_line_level.pkl \
    --seed 42 2>&1 | tee model/jitcl/saved_models_concat_cl/test.log

```

### Ablation Experiment

To train JIT-Contrast without contrasting learning, run the following command:

```shell
python -m JITContrast.concat.run \
    --output_dir=model/jitcl/saved_models_concat/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_train \
    --train_data_file data/jitcl/changes_train.pkl data/jitcl/features_train.pkl \
    --eval_data_file data/jitcl/changes_valid.pkl data/jitcl/features_valid.pkl\
    --test_data_file data/jitcl/changes_test.pkl data/jitcl/features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 24 \
    --eval_batch_size 128 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --feature_size 14 \
    --patience 10 \
    --seed 42 2>&1| tee model/jitcl/saved_models_concat/train.log

```

To obtain the evaluation, run the following command:

```shell
python -m JITFine.concat.run \
    --output_dir=model/jitcl/saved_models_concat/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_test \
    --train_data_file data/jitcl/changes_train.pkl data/jitcl/features_train.pkl \
    --eval_data_file data/jitcl/changes_valid.pkl data/jitcl/features_valid.pkl\
    --test_data_file data/jitcl/changes_test.pkl data/jitcl/features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 256 \
    --eval_batch_size 25 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --only_adds \
    --buggy_line_filepath=data/jitcl/changes_complete_buggy_line_level.pkl \
    --seed 42 2>&1 | tee model/jitcl/saved_models_concat/test.log



To train JIT-Contrast using only semantic feature without contrasting learning, run the following command:

```shell
python -m JITContrast.semantic.run \
    --output_dir=model/jitcl/saved_models_semantic/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_train \
    --train_data_file data/jitcl/changes_train.pkl data/jitcl/features_train.pkl \
    --eval_data_file data/jitcl/changes_valid.pkl data/jitcl/features_valid.pkl\
    --test_data_file data/jitcl/changes_test.pkl data/jitcl/features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 24 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --patience 10 \
    --seed 42 2>&1| tee model/jitcl/saved_models_semantic/train.log
```

To obtain the evaluation, run the following command:

```shell
python -m JITFine.semantic.run \
    --output_dir=model/jitfine/saved_models_semantic/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_test \
    --train_data_file data/jitcl/changes_train.pkl data/jitcl/features_train.pkl \
    --eval_data_file data/jitcl/changes_valid.pkl data/jitcl/features_valid.pkl\
    --test_data_file data/jitcl/changes_test.pkl data/jitcl/features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 24 \
    --eval_batch_size 100 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --patience 10 \
    --seed 42 2>&1|  tee model/jitfine/saved_models_semantic/test.log
```


To train JIT-Contrast using only semantic feature with contrasting learning, run the following command:

```shell
python -m JITContrast.semanticCL.run \
    --output_dir=model/jitcl/saved_models_semantic_cl/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_train \
    --train_data_file data/jitcl/changes_train.pkl data/jitcl/features_train.pkl \
    --eval_data_file data/jitcl/changes_valid.pkl data/jitcl/features_valid.pkl\
    --test_data_file data/jitcl/changes_test.pkl data/jitcl/features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 24 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --patience 10 \
    --seed 42 2>&1| tee model/jitcl/saved_models_semantic_cl/train.log
```

To obtain the evaluation, run the following command:

```shell
python -m JITFine.semanticCL.run \
    --output_dir=model/jitfine/saved_models_semantic_cl/checkpoints \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_test \
    --train_data_file data/jitcl/changes_train.pkl data/jitcl/features_train.pkl \
    --eval_data_file data/jitcl/changes_valid.pkl data/jitcl/features_valid.pkl\
    --test_data_file data/jitcl/changes_test.pkl data/jitcl/features_test.pkl\
    --epoch 50 \
    --max_seq_length 512 \
    --max_msg_length 64 \
    --train_batch_size 24 \
    --eval_batch_size 100 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --patience 10 \
    --seed 42 2>&1|  tee model/jitfine/saved_models_semantic_cl/test.log
```
The fully connected layer and activation function can be adjusted by yourself in the model.py file. The default in the code is two layers plus the Relu activation function.

### **RQ1 Baseline Implementation**

There are 5 baselines in RQ1(i.e., `LApredict`, `Deeper`, `DeepJIT`, `CC2Vec`, and `JITLine`). To reproduce the results of baselines, run the following commands: 

- LApredict

  ```shell
  python -m baselines.LApredict.lapredict
  ```

- Deeper

  ```shell
  python -m baselines.Deeper.deeper
  ```

- DeepJIT

  ```shell
  python -m baselines.DeepJIT.deepjit
  ```

- CC2Vec

  ```shell
  python -m baselines.CC2Vec.cc2vec
  ```

- JITLine

  ```shell
  refer to RQ2
  ```






- For Deeper,  DeepJIT, CC2Vec, we adopt the implementation from this project
  - [*Deep Just-in-Time Defect Prediction: How Far Are We?*](https://github.com/ZZR0/ISSTA21-JIT-DP) 
 
- For JITLine, we adopt the implementation from this project
  - [JITLine: A Simpler, Better, Faster, Finer-grained Just-In-Time Defect Prediction](https://zenodo.org/record/4596503)

- For JITFine, we adopt the implementation from this project
  - [The Best of Both Worlds: Integrating Semantic Features with Expert Features for Defect Prediction and Localization](https://github.com/jacknichao/JIT-Fine)
 
- Special thanks to each work's developers

