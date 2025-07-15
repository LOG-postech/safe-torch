## Reproducing Language Model Experiments

**Pruners:**
*   `magnitude`: Standard magnitude pruning.
*   `wanda`: Pruning based on WANDA (Weight and Activation-based Neural Network Pruning).
*   `sparsegpt`: Pruning based on SparseGPT.
*   `safe`: Our proposed SAFE algorithm.
*   `alps`: Pruning based on ALPS.

We adopted original implementation for competing methods. 


**Example Usage:**

To run an experiment, you can use the following command structure:

```bash
python language/main.py \
    --model <model_name_or_path> \
    --prune_method <method_name> \
    --sparsity_ratio <ratio> \
    --dataset <dataset_name> \
    --nsamples <num_calibration_samples> \
    --seed <random_seed> \
    --seqlen <sequence_length> \
    --batch_size <batch_size_for_safe_opt> \
    --learning_rate <lr_for_safe_opt> \
    --epochs <epochs_for_safe_opt> \
    --lmda <lambda_for_safe_admm> \
    --rho <rho_for_sam_in_safe> \
    --interval <dual_update_interval_for_safe> \
    --activation <True_or_False_for_safe_plus> \
    --wandb <True_or_False_for_wandb_logging> \
    --wandb_project <your_wandb_project_name>
```

**Arguments:**

*   `--model`: Hugging Face model identifier (e.g., `meta-llama/Llama-2-7b-hf`).
*   `--prune_method`: One of the available pruning methods listed above (e.g., `safe`, `wanda`).
*   `--sparsity_ratio`: Target sparsity level (e.g., `0.5` for 50% sparsity).
*   `--sparsity_type`: Type of sparsity (`unstructured`, `2:4`, `4:8`). For N:M sparsity, keep `--sparsity_ratio` fixed to 0.5.
*   `--dataset`: Calibration dataset (`c4`, `wikitext2`).
*   `--nsamples`: Number of calibration samples.
*   `--seed`: Random seed for reproducibility.
*   `--seqlen`: Sequence length for the model.

**SAFE-specific Hyperparameters:**
*   `--batch_size`: Batch size for the SAFE optimization process.
*   `--learning_rate`: Learning rate for the base optimizer within SAFE.
*   `--epochs`: Number of epochs for the SAFE optimization process per block.
*   `--lmda`: Initial penalty parameter for the ADMM component in SAFE.
*   `--rho`: Perturbation size for the SAM component in SAFE.
*   `--interval`: Dual update interval for ADMM in SAFE.
*   `--activation`: Boolean flag to enable SAFE+ (activation-aware projection).
*   `--warmup_epochs`: Warmup epochs for the learning rate scheduler in SAFE.
*   `--accumulation_steps`: Gradient accumulation steps for SAFE.


**Example: Pruning Llama-2-7b with SAFE (50% unstructured sparsity)**
```bash
python language/main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method safe \
    --sparsity_ratio 0.5 \
    --sparsity_type "unstructured" \
    --dataset c4 \
    --nsamples 128 \
    --seed 42 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --epochs 10 \
    --lmda 1e-3 \
    --rho 0.05 \
    --interval 10 \
    --activation True \
    --wandb True \
    --wandb_project "safe-torch-language"
```

More example scripts are available on ./scripts
