import time
import math
from dataclasses import dataclass
from typing import Any, Callable
from abc import ABCMeta, abstractmethod
from functools import partial

from absl import logging
import torch
import torch.nn as nn
from torch.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from .linear_wrapper import LinearPrunerWrapperBase, WandaWrapper, ALPSWrapper, SparseGPTWrapper, WrappedGPT
from .data import get_loaders, TensorData, TensorDataLoader
from .optimizers import SAFE
from .utils import (
    find_layers,
    prepare_calibration_input,
    projection,
    check_sparsity,
    calculate_reconstruction_error,
)


# Prune model with linear wrapper for layer-wise pruners.
# -----------------------------------------------------------------------------

@torch.no_grad()
def prune_model_with_linear_wrapper(
    prune_wrapper_class: LinearPrunerWrapperBase,
    args: Any,
    model: AutoModelForCausalLM,
    dataloader: Any,
    device: torch.device,
    prune_n: int = 0,
    prune_m: int = 0,
    **pruner_extra_args,
):
    """Prune given model."""
    logging.info(f"Starting {prune_wrapper_class.PRUNER_NAME} pruning...")

    logging.info(f"Loading calibration data for {prune_wrapper_class.PRUNER_NAME}.")
    use_cache = getattr(model.config, 'use_cache', None)
    if use_cache is not None:
        model.config.use_cache = False
    
    with torch.no_grad():
        inps, _, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples
        )

    layers = model.model.layers
    current_layer_outputs = torch.zeros_like(inps)
    logging.info(f'{prune_wrapper_class.PRUNER_NAME} calibration data prepared.')

    total_params_count = 0
    total_nnz_count = 0
    overall_pruning_start_time = time.time()

    # Prune layers
    for i in range(len(layers)):
        layer_pruning_start_time = time.time()
        layer = layers[i]
        current_layer_processing_device = device

        layer_key_in_map = f"model.layers.{i}" 
        if hasattr(model, 'hf_device_map') and model.hf_device_map and layer_key_in_map in model.hf_device_map:
            current_layer_processing_device = model.hf_device_map[layer_key_in_map]
            logging.info(f"Layer {i} assigned to device: {current_layer_processing_device} from hf_device_map.")
        
        layer = layer.to(current_layer_processing_device)
        current_inps = inps.to(current_layer_processing_device)
        current_attention_mask = attention_mask.to(current_layer_processing_device) if attention_mask is not None else None
        current_position_ids = position_ids.to(current_layer_processing_device) if position_ids is not None else None

        prunable_submodules_map = find_layers(layer)
        pruner_instances = {}
        logging.info(f'--- Pruning Layer {i} with {prune_wrapper_class.PRUNER_NAME} ---')

        # wrap linear submodules with wrapper/pruner_wrappers
        def add_batch_hook(name):
            def hook_fn(_, inp_tuple, out_tensor):
                actual_input_tensor = inp_tuple[0] if isinstance(inp_tuple, tuple) else inp_tuple
                pruner_instances[name].add_batch(actual_input_tensor.data, out_tensor.data)
            return hook_fn

        handles = []
        for name, module_to_prune in prunable_submodules_map.items():
            pruner_instances[name] = prune_wrapper_class(module_to_prune)
            handles.append(module_to_prune.register_forward_hook(add_batch_hook(name)))

        # Pass calibration data through the layer to populate buffers
        with torch.no_grad():
            for j in range(args.nsamples):
                sample_input = current_inps[j].unsqueeze(0)
                _ = layer(sample_input, attention_mask=current_attention_mask, position_ids=current_position_ids)[0] # Output captured by hooks
        
        for h in handles: # Remove hooks
            h.remove()

        # Perform pruning for each module
        for name, pruner_instance in pruner_instances.items():
            logging.info(f'Applying {prune_wrapper_class.PRUNER_NAME} to layer {i}, module {name}')
            pruner_instance.prune_linear(
                sparsity=args.sparsity_ratio,
                prune_n=prune_n,
                prune_m=prune_m,
                **pruner_extra_args
            )

            weight_data = pruner_instances[name].layer.weight.data
            total_params_count += weight_data.numel()
            total_nnz_count += torch.count_nonzero(weight_data).item()
            
            pruner_instances[name].free()

        # Get outputs from the now-pruned layer
        with torch.no_grad():
            for j in range(args.nsamples):
                sample_input = current_inps[j].unsqueeze(0)
                current_layer_outputs[j] = layer(
                    sample_input,
                    attention_mask=current_attention_mask,
                    position_ids=current_position_ids
                )[0].to('cpu')

        layers[i] = layer.to('cpu')  # Offload layer
        layer_pruning_end_time = time.time()
        logging.info(f'Layer {i} {prune_wrapper_class.PRUNER_NAME} pruning time: {layer_pruning_end_time - layer_pruning_start_time:.2f}s')

        del current_inps, current_attention_mask, current_position_ids, layer, pruner_instances
        torch.cuda.empty_cache()

        inps, current_layer_outputs = current_layer_outputs, inps  # Swap for the next layer (outputs are on CPU)

    overall_pruning_end_time = time.time()
    logging.info(f'Total {prune_wrapper_class.PRUNER_NAME} pruning time: {overall_pruning_end_time - overall_pruning_start_time:.2f}s')
    if total_params_count > 0:
        final_sparsity = 1 - (total_nnz_count / total_params_count)
        logging.info(f'Overall sparsity after {prune_wrapper_class.PRUNER_NAME}: {final_sparsity:.4f} (NNZ: {total_nnz_count}, Total: {total_params_count})')

    if use_cache is not None:
        model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    logging.info(f"{prune_wrapper_class.PRUNER_NAME} pruning finished.")


# Layer-wise pruners (Wanda, SparseGPT, ALPS).
# -----------------------------------------------------------------------------

def prune_wanda(
    args: Any,
    model: AutoModelForCausalLM,
    dataloader: Any,
    device: torch.device,
    prune_n: int = 0,
    prune_m: int = 0
):
    """
    Prunes the model using the WANDA (\|w\|\|x\|_2) method.
    Original code: https://github.com/locuslab/wanda

    Args:
        args: Configuration object with `sparsity_ratio (int)`, `nsamples (int)`, `seed (int)`, `dataset (str)`, `rho (float)` (for ALPS_admm).
        model (AutoModelForCausalLM): The model to prune.
        dataloader (Any): The dataloader for loading calibration data.
        device (torch.device): The default device. Layer-specific devices handled if model.hf_device_map exists.
        prune_n (int): N for N:M structured sparsity (0 for unstructured).
        prune_m (int): M for N:M structured sparsity (0 for unstructured).
    """
    prune_model_with_linear_wrapper(WandaWrapper, args, model, dataloader, device, prune_n, prune_m)
    

def prune_sparsegpt(
    args: Any,
    model: AutoModelForCausalLM,
    dataloader: Any,
    device: torch.device,
    prune_n: int = 0,
    prune_m: int = 0
):
    """
    Prunes the model using the SparseGPT method.
    Original code: https://github.com/IST-DASLab/sparsegpt

    Args:
        args: Configuration object with sparsity_ratio (int), nsamples (int), seed (int), dataset (str).
        model (AutoModelForCausalLM): The model to prune.
        dataloader (Any): The dataloader for loading calibration data.
        device (torch.device): The default device. Layer-specific devices handled if model.hf_device_map exists.
        prune_n (int): N for N:M structured sparsity (0 for unstructured).
        prune_m (int): M for N:M structured sparsity (0 for unstructured).
    """
    percdamp=getattr(args, 'sparsegpt_percdamp', 0.01)
    blocksize=getattr(args, 'sparsegpt_blocksize', 128)
    prune_model_with_linear_wrapper(
        SparseGPTWrapper, args, model, dataloader, device, prune_n, prune_m,
        percdamp=percdamp, blocksize=blocksize
    )
    

def prune_alps(
    args: Any,
    model: AutoModelForCausalLM,
    dataloader: Any,
    device: torch.device,
    prune_n: int = 0,
    prune_m: int = 0
):
    """
    Prunes the model using ALPS method.
    Original code: https://github.com/mazumder-lab/ALPS

    Args:
        args: Configuration object with `sparsity_ratio (int)`, `nsamples (int)`, `seed (int)`, `dataset (str)`, `rho (float)` (for ALPS_admm).
        model (AutoModelForCausalLM): The model to prune.
        dataloader (Any): The dataloader for loading calibration data.
        device (torch.device): The default device. Layer-specific devices handled if model.hf_device_map exists.
        prune_n (int): N for N:M structured sparsity (0 for unstructured).
        prune_m (int): M for N:M structured sparsity (0 for unstructured).
    """
    prune_model_with_linear_wrapper(ALPSWrapper, args, model, dataloader, device, prune_n, prune_m)




@torch.no_grad() 
def prune_safe(
    args,
    model:AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prune_n: int = 0,
    prune_m: int = 0
):
    """
    Prunes the model using SAFE method.
    Paper: https://dummy.com

    Args:
        args: Configuration object with learning_rate (float), lmda (float), rho (float), interval (int), epochs (int).
        model (AutoModelForCausalLM): The model to prune.
        tokenizer (AutoTokenizer): The tokenizer for loading calibration data.
        device (torch.device): The device for computation.
        prune_n (int): N for N:M structured sparsity (0 for unstructured).
        prune_m (int): M for N:M structured sparsity (0 for unstructured).
    """
    logging.info('Starting SAFE pruning...')
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    use_cache = getattr(model.config, 'use_cache', None)
    
    if use_cache is not None:
        model.config.use_cache = False
    
    with torch.no_grad():
        inps, _, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers
    current_layer_outputs = torch.zeros_like(inps) 
    logging.info('SAFE calibration data prepared.')

    # Pruning based on SAFE
    for i in range(len(layers)):
        layer = layers[i].float().to(device) 
        current_inps = inps.float().to(device) 
        
        learning_rate = args.learning_rate
        logging.info(f'Pruning layer {i} with SAFE. Learning rate: {learning_rate}')
        
        # Store outputs of the original (dense) layer to use as targets for optimization
        dense_layer_targets = torch.zeros_like(current_inps, device='cpu') 

        importance_matrix = None
        wrapped_submodules = {} # For collecting activations if args.activation is true
        
        if args.activation: 
            subset_for_act = find_layers(layer)
            importance_matrix = []
            for name in subset_for_act:
                wrapped_submodules[name] = WrappedGPT(subset_for_act[name])

            def add_batch_hook_act(module_name):
                def hook_fn(_, inp, out):
                    actual_input_tensor = inp[0] if isinstance(inp, tuple) else inp
                    wrapped_submodules[module_name].add_batch(actual_input_tensor.data, out.data)
                return hook_fn

            handles_act = []
            for name in wrapped_submodules:
                handles_act.append(subset_for_act[name].register_forward_hook(add_batch_hook_act(name)))
        
        # Get outputs from the dense (original) layer to serve as optimization targets
        with torch.no_grad():
            for j in range(args.nsamples):
                sample_input = current_inps[j].unsqueeze(0) 
                dense_layer_targets[j] = layer(sample_input, attention_mask=attention_mask, position_ids=position_ids)[0].detach().to('cpu')
        
        if args.activation: 
            for h in handles_act:
                h.remove() # Remove hooks

        # Prepare DataLoader for the optimization process
        tensordata = TensorData(current_inps, dense_layer_targets, device) 
        tensordata_loader = TensorDataLoader(tensordata, args.batch_size, shuffle=True, num_workers=0).get_loader()
        
        num_update_steps_per_epoch = len(tensordata_loader)
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(args.epochs * num_update_steps_per_epoch)
        warmup_steps = math.ceil(args.warmup_epochs * num_update_steps_per_epoch)

        if args.activation:
            for name in wrapped_submodules:
                act_scaler = torch.sqrt(wrapped_submodules[name].scaler_row.reshape((1, -1))).to(device)
                importance_matrix.append(act_scaler)
        
        prunable_submodules = find_layers(layer)
        # Collect parameters for the 'safe_params' group and their IDs
        safe_params_list = []
        safe_param_ids = set()
        for name in prunable_submodules:
            # Assuming only 'weight' is part of ADMM for now
            if hasattr(prunable_submodules[name], 'weight') and prunable_submodules[name].weight.requires_grad:
                weight_param = prunable_submodules[name].weight
                safe_params_list.append(weight_param)
                safe_param_ids.add(id(weight_param))


        safe_params_group = {
            'params': safe_params_list,
            'name': 'weights', # Or a more descriptive name like 'admm_weights'
            'admm': True,
        }
        # Collect all other parameters from the layer that are not in safe_params_list
        other_params_list = []
        for param in layer.parameters():
            if param.requires_grad and id(param) not in safe_param_ids:
                other_params_list.append(param)

        other_params_group = {
            'params': other_params_list,
            'name': 'other_params',
            'admm': False,
        }

        # Ensure param_groups only contains groups with actual parameters
        param_groups = []
        if safe_params_group['params']: # Only add if there are params in this group
            param_groups.append(safe_params_group)
        if other_params_group['params']: # Only add if there are params in this group
            param_groups.append(other_params_group)

        opt = SAFE(
            param_groups,
            projection_fn = projection,
            lmda=args.lmda, sparsity=args.sparsity_ratio, interval=args.interval, 
            lr=learning_rate, rho=args.rho, prune_n=prune_n, prune_m=prune_m, 
            importance_matrix=importance_matrix,
            betas=(args.beta1, args.beta2),
        )

        loss_fn = torch.nn.MSELoss().to(device)
        lr_scheduler = get_linear_schedule_with_warmup(
            opt.base_optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
        )
    
        global_step_counter = 0 
        sam_input_cache = []
        # Optimization loop
        for epoch in range(args.epochs): 
            epoch_start_time = time.time()
            total_epoch_loss_items = 0.0 
            
            for batch_inputs, batch_targets in tensordata_loader:
                
                with torch.enable_grad(): 
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = layer(batch_inputs, attention_mask=attention_mask, position_ids=position_ids)[0]
                        loss = loss_fn(outputs, batch_targets)
                        if args.accumulation_steps > 1:
                            loss = loss / args.accumulation_steps
                    loss.backward() 
                    global_step_counter += 1
                    if args.accumulation_steps > 1: 
                        sam_input_cache.append(batch_inputs.detach().clone())


                if global_step_counter % args.accumulation_steps == 0:
                    with torch.enable_grad(): 
                        opt.first_step(zero_grad=True) 
                        
                        if sam_input_cache: 
                            for cached_batch_input in sam_input_cache:
                                with autocast(device_type='cuda', dtype=torch.bfloat16):
                                    loss_sam_step = loss_fn(layer(cached_batch_input, attention_mask=attention_mask, position_ids=position_ids)[0], batch_targets)
                                if args.accumulation_steps > 1:
                                     loss_sam_step = loss_sam_step / args.accumulation_steps
                                loss_sam_step.backward() 
                            sam_input_cache = [] 
                        else: 
                            if args.accumulation_steps == 1: 
                                with autocast(device_type='cuda', dtype=torch.bfloat16):
                                    outputs_perturbed = layer(batch_inputs, attention_mask=attention_mask, position_ids=position_ids)[0]
                                    loss_perturbed = loss_fn(outputs_perturbed, batch_targets)
                                loss_perturbed.backward()


                        opt.second_step(zero_grad=True) 
                        
                    opt.zero_grad(set_to_none=True) 
                
                    total_epoch_loss_items += (loss.item()) * len(batch_inputs) # Accumulate loss items for logging
                    if lr_scheduler is not None:
                        lr_scheduler.step()

            epoch_end_time = time.time()
            avg_epoch_loss = total_epoch_loss_items / len(tensordata_loader.dataset) if len(tensordata_loader.dataset) > 0 else 0.0
            logging.info(f'Layer {i}, Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_epoch_loss}, Time: {epoch_end_time - epoch_start_time:.2f}s')

        # Final projection after optimization
        opt.final_projection()
    
        # Calculate outputs with the now-pruned layer
        with torch.no_grad():
            for j in range(args.nsamples):
                sample_input = current_inps[j].unsqueeze(0) # Inputs are already on device
                current_layer_outputs[j] = layer(sample_input, attention_mask=attention_mask, position_ids=position_ids)[0].to('cpu')

        layers[i] = layer.to('cpu') # Offload layer
        if args.activation:
            del importance_matrix
        del prunable_submodules,opt, lr_scheduler, tensordata, tensordata_loader
        torch.cuda.empty_cache()
        
        inps, current_layer_outputs = current_layer_outputs, inps # Swap for the next layer (outputs are on CPU)

    if use_cache is not None:
        model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    logging.info("SAFE pruning finished.")


@torch.no_grad() 
def prune_magnitude(
    args,
    model:AutoModelForCausalLM,
    device:torch.device,
    prune_n:int=0,
    prune_m:int=0
):
    """
    Prunes the model using the magnitude pruning (\|w\|) method.
    Removes weights with the smallest magnitudes, supporting unstructured or N:M structured sparsity.

    Args:
        args: Configuration object with attribute `sparsity_ratio (int)`.
        model (AutoModelForCausalLM): The model to prune.
        tokenizer (AutoTokenizer): The tokenizer (not directly used here but common signature).
        device (torch.device): The device for computation.
        prune_n (int): N for N:M structured sparsity (0 for unstructured).
        prune_m (int): M for N:M structured sparsity (0 for unstructured).
    """
    logging.info("Starting magnitude pruning...")
    layers = model.model.layers 
    # Pruning based on magnitude
    for i in range(len(layers)):
        layer = layers[i].to(device)
        subset = find_layers(layer) 

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)

            if prune_n != 0 and prune_m != 0: # N:M structured sparsity
                W_mask = torch.zeros_like(W, dtype=torch.bool) 
                for col_chunk_idx in range(W_metric.shape[1] // prune_m):
                    start_col = col_chunk_idx * prune_m
                    end_col = start_col + prune_m
                    tmp_metric_chunk = W_metric[:, start_col:end_col]
                    
                    _, topk_indices = torch.topk(tmp_metric_chunk, prune_n, dim=1, largest=False)
                    
                    W_mask[:, start_col:end_col].scatter_(1, topk_indices, True)
            else: # Unstructured sparsity
                num_elements_to_prune = int(W.numel() * args.sparsity_ratio)
                threshold = torch.kthvalue(W_metric.flatten(), num_elements_to_prune + 1).values 
                W_mask = (W_metric <= threshold)

            W[W_mask] = 0 
        
        layers[i] = layer.to('cpu') 
        torch.cuda.empty_cache()
    logging.info("Magnitude pruning finished.")
