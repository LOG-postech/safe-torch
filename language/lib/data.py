from absl import logging
import numpy as np
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

### Code adopted from Shin et al. (2024), Rethinking Pruning Large Language Models: Benefits and Pitfalls of Reconstruction Error Minimization.
### https://github.com/LOG-postech/rethinking-LLM-pruning

class TensorData(torch.utils.data.Dataset):
    def __init__(self, data, targets, device):
        self.data = data
        self.targets = targets
        self.device = device

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x.to(self.device), y.to(self.device)

    def __len__(self):
        return len(self.targets)    

class TensorData_infer(torch.utils.data.Dataset):
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __getitem__(self, index):
        x = self.data[index]
        return x.to(self.device)

    def __len__(self):
        return len(self.data)    

class TensorDataLoader:
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def get_loader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=False
        )

# Code adopted from https://github.com/locuslab/wanda

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self,
                input_ids: torch.Tensor
                ):
        self.input_ids = input_ids

def get_wikitext2(
    nsamples: int,
    seed: int,
    seqlen: int,
    tokenizer: AutoTokenizer
) -> tuple[list[tuple[torch.tensor, torch.Tensor]],list[torch.Tensor]]:
    """
    Load and process the wikitext2 dataset. Preprocessing logic adopted from sparseGPT.
    Args:
        nsamples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for the input data.
        tokenizer (AutoTokenizer): Tokenizer to use for encoding the data.
    Returns:
        tuple: A tuple containing the training data loader and the test data.
            The training data loader is a list of tuples, each containing input tensor and target tensors. 
            The test data is a tensor of encoded text.
    """
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train',trust_remote_code=True)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test',trust_remote_code=True)

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        if 'Gemma' in tokenizer.__class__.__name__:
            inp[:,0] = tokenizer.bos_token_id
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(
    nsamples: int,
    seed: int,
    seqlen: int,
    tokenizer: AutoTokenizer
)-> tuple[list[tuple[torch.tensor, torch.Tensor]],list[torch.Tensor]]:
    """
    Load and process the C4 dataset. Preprocessing logic adopted from sparseGPT.
    Args:
        nsamples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for the input data.
        tokenizer (AutoTokenizer): Tokenizer to use for encoding the data.
    Returns:
        tuple: A tuple containing the training data loader and the test data.
            The training data loader is a list of tuples, each containing input tensor and target tensors. 
            The test data is a tensor of encoded text.
    """
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train',trust_remote_code=True,cache_dir='~/.datacache/huggingface/datasets')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation',trust_remote_code=True,cache_dir='~/.datacache/huggingface/datasets')
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for i in range(nsamples):
        logging.info(f'Constructing sample {i}')
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_loaders(
    name: str, 
    nsamples: int = 128,
    seed: int = 0,
    seqlen: int = 2048,
    tokenizer: AutoTokenizer = None
) -> tuple[list[tuple[torch.tensor, torch.Tensor]], list[torch.Tensor]]:
    """
    Load and process the specified dataset.
    Args:
        name (str): Name of the dataset to load. (c4,wikitext2)
        nsamples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for the input data.
        tokenizer (AutoTokenizer): Tokenizer to use for encoding the data.
    Returns:
        tuple: A tuple containing the training data loader and the test data.
            The training data loader is a list of tuples, each containing input tensor and target tensors. 
            The test data is a tensor of encoded text.
    """
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:            
        return get_c4(nsamples, seed, seqlen, tokenizer)
