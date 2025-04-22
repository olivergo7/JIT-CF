from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel)
from tqdm import tqdm, trange
import multiprocessing
from JITCF.concatCL.contrastive_loss import ContrastiveLoss
from JITCF.semantic.model import Model
from JITCF.my_util import convert_examples_to_features, TextDataset, eval_result, create_path_if_not_exist

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 5
    args.warmup_steps = 0
    model.to(args.device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    contrastive_criterion = ContrastiveLoss(temperature=0.5).to(args.device)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", args.max_steps)

    best_f1 = 0
    global_step = 0
    model.zero_grad()
    patience = 0

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_loss = 0
        tr_num = 0
        for step, batch in enumerate(bar):
            (inputs_ids, attn_masks,  labels) = [x.to(args.device) for x in batch]
            model.train()
            labels = labels.cpu()
            loss_cls, logits, _, features = model(inputs_ids, attn_masks, labels)
            loss_contrastive = contrastive_criterion(features, labels)
            loss = loss_cls + loss_contrastive

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % args.save_steps == 0:
                logger.info("epoch {} step {} loss {:.5f}".format(idx, step + 1, tr_loss / tr_num))
                tr_loss = 0
                tr_num = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            global_step += 1
            if (step + 1) % args.save_steps == 0:
                results = evaluate(args, model, tokenizer, eval_when_training=True)
                checkpoint_prefix = f'epoch_{idx}_step_{step}'
                output_dir = os.path.join(args.output_dir, checkpoint_prefix)
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': idx,
                    'step': step,
                    'patience': patience,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}, os.path.join(output_dir, 'model.bin'))
                logger.info("Saved checkpoint to %s", output_dir)

                if results['eval_f1'] > best_f1:
                    best_f1 = results['eval_f1']
                    logger.info("  Best f1: %.4f", best_f1)
                    best_dir = os.path.join(args.output_dir, 'checkpoint-best-f1')
                    os.makedirs(best_dir, exist_ok=True)
                    torch.save({
                        'epoch': idx,
                        'step': step,
                        'patience': patience,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}, os.path.join(best_dir, 'model.bin'))
                    logger.info("Saved best model to %s", best_dir)
                    patience = 0
                else:
                    patience += 1
                    if patience > args.patience * 5:
                        logger.info('Early stopping triggered at patience %d', patience)
                        return


def evaluate(args, model, tokenizer, eval_when_training=False):
    # build dataloader
    cache_dataset = os.path.join(os.path.dirname(args.eval_data_file[0]),
                                 f'valid_set_cache_msg{args.max_msg_length}.pkl')
    if os.path.exists(cache_dataset):
        eval_dataset = pickle.load(open(cache_dataset, 'rb'))
    else:
        eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file, mode='valid')
        pickle.dump(eval_dataset, open(cache_dataset, 'wb'))
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (inputs_ids, attn_masks, _, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss, logit = model(inputs_ids, attn_masks, labels)
            eval_loss += loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    y_preds = logits[:, -1] > best_threshold
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='binary')
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, tokenizer, best_threshold=0):
    # build dataloader
    cache_dataset = os.path.join(os.path.dirname(args.test_data_file[0]),
                                 f'test_set_cache_msg{args.max_msg_length}.pkl')
    if os.path.exists(cache_dataset):
        eval_dataset = pickle.load(open(cache_dataset, 'rb'))
    else:
        eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file, mode='test')
        pickle.dump(eval_dataset, open(cache_dataset, 'wb'))
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (inputs_ids, attn_masks, _, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss, logit = model(inputs_ids, attn_masks, labels)
            eval_loss += loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # output result
    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)

    y_preds = logits[:, -1] > best_threshold
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='binary')
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    result = []
    for example, pred, prob in zip(eval_dataset.examples, y_preds, logits[:, -1]):
        result.append([example.commit_id, prob, pred, example.label])
    RF_result = pd.DataFrame(result)
    RF_result.to_csv(os.path.join(args.output_dir, "predictions.csv"), sep='\t', index=None)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", nargs=2, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--eval_data_file", nargs=2, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", nargs=2, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_seed', type=int, default=123456,
                        help="random seed for data order initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    parser.add_argument('--feature_size', type=int, default=14,
                        help="Number of features")
    parser.add_argument('--num_labels', type=int, default=2,
                        help="Number of labels")
    parser.add_argument('--semantic_checkpoint', type=str, default=None,
                        help="Best checkpoint for semantic feature")
    parser.add_argument('--manual_checkpoint', type=str, default=None,
                        help="Best checkpoint for manual feature")
    parser.add_argument('--max_msg_length', type=int, default=64,
                        help="Number of labels")
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for early stop')
    parser.add_argument("--only_adds", action='store_true',
                        help="Whether to run eval on the only added lines.")
    parser.add_argument("--buggy_line_filepath", type=str,
                        help="complete buggy line-level  data file for RQ3")
    args = parser.parse_args()
    create_path_if_not_exist(args.output_dir)
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )

    # Set seed
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

    # add new special tokens
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ["[ADD]", "[DEL]"]})

    model = RobertaModel.from_pretrained(args.model_name_or_path, config=config)

    model = Model(model, config, tokenizer, args)

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        logger.info("Creating features from index file at %s ", args.train_data_file)
        cache_dataset = os.path.join(os.path.dirname(args.test_data_file[0]),
                                     f'train_set_cache_msg{args.max_msg_length}.pkl')
        if os.path.exists(cache_dataset):
            train_dataset = pickle.load(open(cache_dataset, 'rb'))
        else:
            train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file)
            pickle.dump(train_dataset, open(cache_dataset, 'wb'))

        for idx, example in enumerate(train_dataset.examples[:1]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(idx))
            logger.info("label: {}".format(example.label))
            logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
            logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        checkpoint = torch.load(output_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)

        result = evaluate(args, model, tokenizer)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        checkpoint = torch.load(output_dir)
        logger.info("Successfully load epoch {}'s model checkpoint".format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.to(args.device)

        test(args, model, tokenizer, best_threshold=0.5)
        eval_result(os.path.join(args.output_dir, "predictions.csv"), args.test_data_file[-1])

    return results


if __name__ == "__main__":
    main()
