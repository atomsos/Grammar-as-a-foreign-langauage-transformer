import os
import random
import platform


import wandb
import torch
from tokenizers import Tokenizer

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tokenizer import get_tokenizer_wordlevel, get_tokenizer_bpe

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

system = platform.system()
if system == 'Linux':
    NUM_RPOC = 16
elif system == 'Windows':
    NUM_PROC = None
elif system == 'Darwin':
    NUM_PROC = 4




def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)


def pad_collate_fn(batch):
    src_sentences, trg_sentences = [], []
    for sample in batch:
        src_sentences += [sample[0]]
        trg_sentences += [sample[1]]

    src_sentences = pad_sequence(
        src_sentences, batch_first=True, padding_value=0)
    trg_sentences = pad_sequence(
        trg_sentences, batch_first=True, padding_value=0)

    return src_sentences, trg_sentences


class TranslationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_encoded = self.dataset[idx]['words']
        trg_encoded = self.dataset[idx]['parse_tree']

        return (
            torch.tensor(src_encoded),
            torch.tensor(trg_encoded),
        )


class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):

        # Dataset is already sorted so just chunk indices
        # into batches of indices for sampling
        self.batch_size = batch_size
        self.indices = range(len(dataset))
        self.batch_of_indices = list(chunk(self.indices, self.batch_size))
        self.batch_of_indices = [batch.tolist()
                                 for batch in self.batch_of_indices]

    def __iter__(self):
        random.shuffle(self.batch_of_indices)
        return iter(self.batch_of_indices)

    def __len__(self):
        return len(self.batch_of_indices)


def get_data(example_cnt):
    # data = load_dataset('conll2012_ontonotesv5','english_v4',split='train').shuffle(seed=42)
    # data['sentences'] = sum(data['sentences'], [])

    data = load_dataset("json", data_files="data.json.gz", split='train')
    keys = ['part_id', 'pos_tags', 'predicate_lemmas', 'predicate_framenet_ids', 'word_senses', 'speaker', 'named_entities', 'srl_frames', 'coref_spans']
    data = data.remove_columns(keys)
    # data = load_dataset("csv", data_files="data.csv", split='train')
    if example_cnt and example_cnt < len(data):
        data = data.select(range(example_cnt))
    # data = data.rename_column('translation.de', 'parse_tree')
    # data = data.rename_column('translation.en', 'words')

    ## filter None
    def filter_None(examples):
        return [_ != None for _ in examples['parse_tree']]
    data = data.filter(filter_None, batched=True, batch_size=5000)

    return data


def preprocess_data(data, tokenizer, max_seq_len, test_proportion):

    # Tokenize
    def tokenize(examples):
        return {
            'words': [tokenizer.encode(' '.join(words)).ids for words in examples['words']],
            'sentence': [' '.join(words) for words in examples['words']],
            'parse_tree': [tokenizer.encode(parse_tree.replace('(', ' ( ').replace(')', ' ) ').strip()
                ).ids for parse_tree in examples['parse_tree']],
        }
    data = data.map(tokenize, batched=True, batch_size=10000, num_proc=NUM_PROC)

    # Compute sequence lengths
    def sequence_length(example):
        return {
            'length_src': [len(item) for item in example['words']],
            'length_trg': [len(item) for item in example['parse_tree']],
        }
    data = data.map(sequence_length, batched=True,
                    batch_size=10000, num_proc=NUM_PROC)

    # Test unknown
    UNK = tokenizer.token_to_id('[UNK]')

    def get_num_unknown(items):
        return {
            'unk_src': [_.count(UNK) for _ in items['words']],
            'unk_trg': [_.count(UNK) for _ in items['parse_tree']],
            'tot_src': [len(_) for _ in items['words']],
            'tot_trg': [len(_) for _ in items['parse_tree']],
        }

    unknowns = data.map(get_num_unknown, batched=True,
                        batch_size=10000, num_proc=NUM_PROC)
    ks = list(map(lambda x: sum(unknowns[x]), [
              'unk_src', 'tot_src', 'unk_trg', 'tot_trg']))
    u_src, u_trg = ks[0]/ks[1], ks[2]/ks[3]
    total_token = sum(data['length_src'])
    # log.info("unknown ratio: src: {}%, trg: {}%".format(u_src*100, 3), round(ks[2]/ks[3]*100, 3))
    _log = {
        'Val': {
            'Real_Vocab_size': tokenizer.get_vocab_size(),
            'Unknown_ratio_src': u_src,
            'Unknown_ratio_trg': u_trg,
            'Total_Token': total_token,
        }
    }
    log.info(_log)
    if wandb.run is not None:
        wandb.log(_log)
    # Filter by sequence lengths

    def filter_long(examples):
        return [l_src <= max_seq_len for l_src, l_trg in zip(examples['length_src'], examples['length_trg'])]
        # return [l_src <= max_seq_len and l_trg <= max_seq_len for l_src, l_trg in zip(examples['length_src'], examples['length_trg'])]
    data = data.filter(filter_long, batched=True,
                       batch_size=10000, num_proc=NUM_PROC)

    # Split
    data = data.train_test_split(test_size=test_proportion)

    # Sort each split by length for dynamic batching (see CustomBatchSampler)
    log.info("Start sorting")
    data['train'] = data['train'].sort(
        'length_src', reverse=True, writer_batch_size=100000)
    log.info("Training sort done")
    data['test'] = data['test'].sort(
        'length_src', reverse=True, writer_batch_size=100000)
    log.info("Testing sort done")

    return data


def get_translation_dataloaders(
    dataset_size,
    vocab_size,
    tokenizer_type,
    tokenizer_save_pth,
    test_proportion,
    batch_size,
    max_seq_len,
    report_summary,
):

    data = get_data(dataset_size)

    if tokenizer_type == 'wordlevel':
        tokenizer = get_tokenizer_wordlevel(data, vocab_size)
    elif tokenizer_type == 'bpe':
        tokenizer = get_tokenizer_bpe(data, vocab_size)

    # Save tokenizers
    try:
        tokenizer.save(tokenizer_save_pth)
    except:
        tokenizer = Tokenizer.from_file(tokenizer_save_pth)

    data = preprocess_data(data, tokenizer, max_seq_len, test_proportion)

    data_dir = "data/data-{}-{}".format(len(data), tokenizer_type)
    data.save_to_disk(data_dir)

    if report_summary:
        wandb.run.summary['train_len'] = len(data['train'])
        wandb.run.summary['val_len'] = len(data['test'])

    # Create pytorch datasets
    train_ds = TranslationDataset(data['train'])
    val_ds = TranslationDataset(data['test'])

    # Create a custom batch sampler
    custom_batcher_train = CustomBatchSampler(train_ds, batch_size)
    custom_batcher_val = CustomBatchSampler(val_ds, batch_size)

    # Create pytorch dataloaders
    train_dl = DataLoader(train_ds, collate_fn=pad_collate_fn,
                          batch_sampler=custom_batcher_train, pin_memory=True)
    val_dl = DataLoader(val_ds, collate_fn=pad_collate_fn,
                        batch_sampler=custom_batcher_val, pin_memory=True)

    return train_dl, val_dl


if __name__ == '__main__':
    from config import configs
    config = configs['unofficial_single_gpu_config']
    dataset_size = config['DATASET_SIZE']
    vacab_size = config['VOCAB_SIZE']
    tokenizer_type = config['tokenizer_type'.upper()]
    train_dl, val_dl = get_translation_dataloaders(
        dataset_size=dataset_size,
        vocab_size=vacab_size,
        tokenizer_type=tokenizer_type,
        tokenizer_save_pth="runs/test/tokenizer.json",
        test_proportion=0.01,
        batch_size=200,
        max_seq_len=40,
        report_summary=False,
    )
