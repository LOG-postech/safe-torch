from absl import logging
from collections import defaultdict
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import fnmatch
from .data import get_loaders 

# Code adopted from https://github.com/locuslab/wanda

def eval_ppl(
    args,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device = torch.device("cuda:0")
) -> dict:
    """
    Evaluate the model on the wikitext2 and c4 datasets.
    Args:
        args: Namespace, command line arguments.
        model (AutoModelForCausalLM): The model to evaluate.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the data.
        device (torch.device): The device to use for evaluation.
    Returns:
        dict: A dictionary containing the perplexity (ppl) for each dataset.
    """
    dataset = ["wikitext2", "c4"]
    ppls = defaultdict(float)
    for d in dataset:
        # Print status
        logging.info(f"evaluating on {d}")

        # Get the test loader
        _, testloader = get_loaders(
            d, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer 
        )
        # Evaluate ppl in no grad context to avoid updating the model
        with torch.no_grad():
            ppl_test = calculate_ppl(model, testloader, 1, device)
            ppls[d] = ppl_test
    return ppls 

def calculate_ppl(
    model: AutoModelForCausalLM,
    testenc,
    bs: int = 1,
    device: torch.device = None
) -> float:
    """
    Calculate the perplexity of the model on the test set.
    Args:
        model (AutoModelForCausalLM): The model to evaluate.
        testenc: The test set encoded as input IDs. Must have input_ids attribute (e.g. TokenizerWrapper,BatchEncoding).
        bs (int): Batch size for evaluation.
        device (torch.device): The device to use for evaluation.
    Returns:
        float: The perplexity of the model on the test set.
    """
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    logging.info(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            logging.info(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen) 
        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()
