import torch
import torch.nn as nn
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

def projection(
    w: list[torch.Tensor],
    sparsity: float,
    prune_n: int =0,
    prune_m: int = 0,
    importance_matrix: list[torch.Tensor] = None
) -> list[torch.Tensor]:
    """
    Args:
        w (list[torch.Tensor]): list of weights (nxm) to be projected
        sparsity (float): target sparsity
        prune_n (int): n for n:m semi-structured sparsity
        prune_m (int): m for n:m semi-structured sparsity
        importance_matrix (list[torch.Tensor], optional): importance matrix (diag(mxm)) or vector (1xm) for each weight
    Returns:
        new_zs (list[torch.Tensor]): list of projected weights
    """
    
    new_zs = []
    if importance_matrix is not None: # Generalized projection, SAFE+
        for weight,a in zip(w,importance_matrix):
            new_z = weight.data.clone().detach()
            z_metric = torch.abs(weight) * a
            if prune_n != 0: # n:m semi-structured sparsity
                z_mask = (torch.zeros_like(new_z)==1)
                for ii in range(z_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = z_metric[:,ii:(ii+prune_m)].float()
                        z_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else: # unstructured sparsity
                thresh = torch.sort(z_metric.flatten().cuda())[0][int(new_z.numel()*sparsity)].cpu()
                z_mask = (z_metric<=thresh)
            new_z[z_mask] = 0
            new_zs.append(new_z)
    else: # Standard projection, SAFE
        for weight in w:
            new_z = weight.data.clone().detach()
            z_metric = torch.abs(weight)
            if prune_n != 0: # n:m semi-structured sparsity
                z_mask = (torch.zeros_like(new_z)==1)
                for ii in range(z_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = z_metric[:,ii:(ii+prune_m)].float()
                        z_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else: # unstructured sparsity
                thresh = torch.sort(z_metric.flatten().cuda())[0][int(new_z.numel()*sparsity)].cpu()
                z_mask = (z_metric<=thresh)
            new_z[z_mask] = 0
            new_zs.append(new_z)
    return new_zs
 
def find_layers(
    module: nn.Module,
    layers: list = [nn.Linear],
    name: str = ''
) -> dict:
    """
    Recursively find the layers of a certain type in a module.
    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find. 
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model:AutoModelForCausalLM)-> float:
    """
    Calculates and logs the sparsity of each layer and the total model sparsity.
    Args:
        model: The model to check (AutoModelForCausalLM).
    Returns:
        float: The total sparsity of the model.
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        logging.info(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(
    model:AutoModelForCausalLM,
    dataloader:torch.utils.data.DataLoader,
    device:torch.device,
    nsamples:int=128
)-> tuple:
    """
    Prepare input data for model calibration.
    Supports OpenLM models and HF models (Llama2, Llama3, Gemma2).
    Offloads most of the model to CPU, loading only necessary parts to the device to maximize memory efficiency.
    Captures the activations for the first transformer layer's input.
    Args:
        model (AutoModelForCausalLM): The model for which to prepare calibration data.
        dataloader (torch.utils.data.DataLoader): DataLoader providing calibration data.
        device (torch.device): The device to use for capturing activations.
        nsamples (int): Number of samples to prepare.

    Returns:
        tuple: (inps, outs, attention_mask, position_ids)
            inps (torch.Tensor): Input activations to the first transformer block.
            outs (torch.Tensor): Placeholder for outputs (same shape as inps).
            attention_mask (torch.Tensor): Attention mask from calibration data.
            position_ids (torch.Tensor): Position IDs from calibration data.
    """
    use_cache = getattr(model.config, 'use_cache', None)
    if use_cache is not None:
        model.config.use_cache = False
    
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    if hasattr(model.model,'rotary_emb'): # Gemma does not have rotary_emb
        model.model.rotary_emb = model.model.rotary_emb.to(device)
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    
    hidden_size = getattr(model.config, 'hidden_size', None)
    if hidden_size is None:
        hidden_size = getattr(model.config, 'dim', None)
        if hidden_size is None:
            raise ValueError("Could not find hidden_size or dim in model config")
    
    if not (hasattr(model, 'model') and hasattr(model.model, 'layers')):
        raise ValueError("Could not find model.model.layers in the model structure")
    
    inps = torch.zeros((nsamples, model.seqlen, hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        # Helper module to catch inputs to the first transformer layer
        def __init__(self, module):
            super().__init__()
            self.module = module
        
        def forward(self, inp, **kwargs):
            if cache['i'] < nsamples: 
                 inps[cache['i']] = inp.detach() 
            cache['i'] += 1
            if 'attention_mask' in kwargs:
                cache['attention_mask'] = kwargs['attention_mask']
            if 'position_ids' in kwargs: 
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError 
    
    original_first_layer = layers[0]
    layers[0] = Catcher(layers[0]) 
    
    samples_collected = 0
    for batch in dataloader:
        if samples_collected >= nsamples:
            break
        try:
            model(batch[0].to(device))
        except ValueError: # Expected exception from Catcher
            pass 
        samples_collected = min(cache['i'], nsamples)


    layers[0] = original_first_layer # Restore original layer
    
    # Offload parts from device
    layers[0] = layers[0].to('cpu')
    model.model.embed_tokens = model.model.embed_tokens.to('cpu')
    model.model.norm = model.model.norm.to('cpu')
    if hasattr(model.model,'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to('cpu')

    # Finalize outputs
    # If fewer than nsamples were collected, slice inps
    if samples_collected < nsamples:
        logging.warning(f"Collected {samples_collected} samples, less than requested {nsamples}.")
        inps = inps[:samples_collected]
    
    inps = inps.to('cpu') # Move collected inputs to CPU
    outs = torch.zeros_like(inps) # Placeholder for outputs
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    if use_cache is not None:
        model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    return inps, outs, attention_mask, position_ids

def calculate_reconstruction_error(
    inps:torch.Tensor,
    outs:torch.Tensor,
    device:torch.device
)-> float:
    """
    Calculates the mean squared error between inputs and outputs in FP32.

    Args:
        inps (torch.Tensor): Input tensors.
        outs (torch.Tensor): Output tensors.
        device (torch.device): Device to use for calculation (though result is CPU).

    Returns:
        float: The reconstruction error (MSE) on CPU.
    
    Note:
        MSE calculation on GPU can be faster, but this function returns a CPU scalar.
        It's typically not a bottleneck if called infrequently (e.g., per block).
    """
    with torch.no_grad():
        inps_device = inps.to(device, non_blocking=True)
        outs_device = outs.to(device, non_blocking=True)
        return torch.nn.MSELoss()(inps_device.float(), outs_device.float()).cpu().item()
    