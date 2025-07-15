import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.prune import prune_safe, prune_alps, prune_wanda, prune_magnitude, prune_sparsegpt, prune_sam_imp
from lib.eval import eval_ppl
from lib.utils import check_sparsity
from absl import logging, app, flags
from importlib.metadata import version
from argparse import Namespace
from lib.data import get_loaders, TensorData, TensorDataLoader

#logging
import wandb

logging.info(f"{version('torch')=}")
logging.info(f"{version('transformers')=}")
logging.info(f"{version('accelerate')=}")
logging.info(f'# of gpus: {torch.cuda.device_count()}')

FLAGS = flags.FLAGS

def get_llm(
    model_name:str, 
    seqlen:int=2048
)-> AutoModelForCausalLM:
    """
    Load the model from huggingface hub or local directory.
    The model should be a causal language model, such as Llama2, Gemma, etc.
    Args:
        Model_name: str, directly from huggingface hub, or the directory of the model.
        seqlen: int, the maximum sequence length for the model.
    Returns:
        model: AutoModelForCausalLM, the model loaded from huggingface hub.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code = True,
        use_flash_attention_2=False
    )
    assert seqlen<=model.config.max_position_embeddings, f"seqlen({seqlen}) should be less than or equal to model.config.max_position_embeddings({model.config.max_position_embeddings})"
    model.seqlen = seqlen
    return model

def main(argv):
    global FLAGS
    arguments = FLAGS.flag_values_dict() 
    
    ## delete afterwards
    if FLAGS.wandb:
        wandb.init(project=FLAGS.wandb_project)

        if not dict(wandb.config):  
            wandb.config.update(arguments)  
        else: 
            updated_args = {
                k: wandb.config.get(k, v) for k, v in arguments.items()
            }
            FLAGS = Namespace(**updated_args)
            logging.info(f"Updated args with wandb.config: {FLAGS}")
    else:
        logging.info('\n' + '\n'.join([f'{k} = {v}' for k, v in arguments.items()]))
        
    
    # Setting seeds for reproducibility
    np.random.seed(FLAGS.seed)
    torch.random.manual_seed(FLAGS.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if FLAGS.sparsity_type != "unstructured":
        assert FLAGS.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, FLAGS.sparsity_type.split(":"))

    logging.info(f"loading llm model {FLAGS.model}")
    model = get_llm(FLAGS.model,FLAGS.seqlen)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model, use_fast=False)
    
    dataloader, _ = get_loaders(
        FLAGS.dataset,
        nsamples=FLAGS.nsamples,
        seed=FLAGS.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer
    )

    device = torch.device("cuda:0")
    if "30b" in FLAGS.model or "65b" in FLAGS.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    logging.info(f"use device {device}")
    if FLAGS.sparsity_ratio != 0:
        logging.info("pruning starts")
        if FLAGS.prune_method == "magnitude":
            prune_magnitude(FLAGS, model, device, prune_n=prune_n, prune_m=prune_m)
        elif FLAGS.prune_method == "wanda":
            prune_wanda(FLAGS, model, dataloader, device, prune_n=prune_n, prune_m=prune_m)
        elif FLAGS.prune_method == "sparsegpt":
            prune_sparsegpt(FLAGS, model, dataloader, device, prune_n=prune_n, prune_m=prune_m)
        elif FLAGS.prune_method == "alps":
            prune_alps(FLAGS, model, dataloader, device, prune_n=prune_n, prune_m=prune_m)
        elif FLAGS.prune_method == "safe":
            prune_safe(FLAGS, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif FLAGS.prune_method == "sam_imp":
            prune_sam_imp(FLAGS, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        
    logging.info("pruning finished")
    model = model.to(torch.float16)
    model.seq_len = FLAGS.seqlen
    model = model.to(device)

    # sparsity sanity check
    ################################################################
    logging.info("*"*30)
    sparsity_ratio = check_sparsity(model)
    logging.info(f"sparsity sanity check {sparsity_ratio:.4f}")
    logging.info("*"*30)
    ################################################################
    
    # perplexity evaluation
    ppl_test = eval_ppl(FLAGS, model, tokenizer, device)
    logging.info([(key,ppl) for key,ppl in ppl_test.items()])
    if FLAGS.wandb:
        wandb.log({"sparsity_ratio": sparsity_ratio, **{f"ppl_test({key})": value for key, value in ppl_test.items()}})

if __name__ == '__main__':
    flags.DEFINE_string('model', 'meta-llama/Llama-2-7b-hf', 'model to prune.')
    flags.DEFINE_integer('seqlen', 2048, 'Sequence length for the model.')
    flags.DEFINE_integer('seed', 0, 'Seed for sampling the calibration data.')
    flags.DEFINE_integer('nsamples', 128, 'Number of calibration samples.')
    flags.DEFINE_float('sparsity_ratio', 0.5, 'Sparsity level')
    flags.DEFINE_enum('sparsity_type', "unstructured", ["unstructured", "4:8", "2:4"], 'Type of sparsity.')
    flags.DEFINE_enum('prune_method', "safe", ["magnitude", "wanda", "sparsegpt", "safe", "alps","sam-imp"], 'Pruning method.')
    flags.DEFINE_enum('dataset', 'c4', ["c4", "wikitext2"], 'Calibration dataset.')
    
    # SAFE hyperparams
    flags.DEFINE_float('lmda', 1e-3, 'Penalty parameter for SAFE dual update.')
    flags.DEFINE_integer('batch_size', 8, 'Batch size for SAFE.')
    flags.DEFINE_float('learning_rate', 2e-4, 'Learning rate for SAFE.')
    flags.DEFINE_integer('epochs', 30, 'Number of epochs for SAFE.')
    flags.DEFINE_integer('interval', 32, 'Dual update interval for SAFE.')
    flags.DEFINE_integer('warmup_epochs', 2, 'Warmup epochs for SAFE.')
    flags.DEFINE_integer('accumulation_steps', 1, 'Accumulation steps for SAFE')
    flags.DEFINE_bool('activation', False, 'Activation based sparsity, SAFE+.')
    flags.DEFINE_float('rho', 2e-4, 'Rho for SAM.')
    flags.DEFINE_float('beta1', 0.9, 'Beta1 for Adam.')
    flags.DEFINE_float('beta2', 0.95, 'Beta2 for Adam.')

    flags.DEFINE_bool('eval_zero_shot', True, 'Whether to evaluate zero-shot performance.')
    flags.DEFINE_bool('wandb', False, 'Whether to use wandb for logging.')
    flags.DEFINE_string('wandb_project', 'safe-torch', 'wandb project name.')
    
    app.run(main)
