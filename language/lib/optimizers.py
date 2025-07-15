import torch

class SAFE(torch.optim.Optimizer):
    def __init__(
        self, 
        param_groups,
        projection_fn,
        sparsity: float,
        interval: int,
        lmda: float = 1e-3,
        lr: float = 2e-4,
        rho: float = 0.05,
        prune_n: int = 0,
        prune_m: int = 0,
        importance_matrix: list[torch.Tensor] = None,
        **kwargs
    ):
        """
        SAFE optimizer 
        Args:
            param_groups (list): List of parameter groups.
                Each group is a dict, e.g., {'params': [...], 'admm': True, 'lmda': 0.01}
                'admm': True indicates this group's params are subject to ADMM. Dual and split variables are created as optimizer state for these params.
                'lmda': Penalty parameter for this ADMM group.
            projection_fn (callable): Projection function to use. Should take a list of tensors and return a list of projected tensors.
                Expected signature: projection_fn(params_list, sparsity, prune_n, prune_m, importance_matrix) -> projected_params_list
            sparsity (float): Sparsity target.
            interval (int): Interval for dual update.
            lmda (float): penalty parameter.
            lr (float): Learning rate for the base optimizer.
            rho (float): Perturbation size for SAM.
            prune_n (int): n for n:m structured sparsity.
            prune_m (int): m for n:m structured sparsity.
            importance_matrix (list[torch.Tensor], optional): Importance matrix used for generalized projection. Must have the same structure as param_groups with admm=True.
            **kwargs: Additional arguments for the base optimizer.
        """
        if not callable(projection_fn):
            raise TypeError("projection_fn must be a callable function.")
        self.projection= projection_fn
        processed_param_groups = []
        for i, group in enumerate(param_groups):
            if group.get('admm', False):
                if not group['params']: # Should not happen if group is valid
                    print(f"Warning: ADMM group {i} has no params.")
                    processed_param_groups.append(group)
                    continue
                admm_params_list = group['params'] # This should be a list of tensors

                group['duals'] = [torch.zeros_like(p, device=p.device) for p in admm_params_list]
                if importance_matrix is not None:
                    if len(importance_matrix) != len(admm_params_list):
                        raise ValueError(f"importance_matrix must have the same length as params in group {i}.")
                group['splits'] = self.projection(admm_params_list, sparsity, prune_n, prune_m, importance_matrix)

                if 'lmda' not in group:
                    group['lmda'] = lmda
            processed_param_groups.append(group)
        
        defaults = dict(lr=lr, rho=rho, **kwargs) # lmda is now per-group
        super(SAFE, self).__init__(processed_param_groups, defaults)
        sam_param_groups = []
        for pg in self.param_groups: # self.param_groups is now processed_param_groups
            sam_pg = {k: v for k, v in pg.items() if k not in ['duals', 'splits', 'admm', 'lmda']}
            sam_param_groups.append(sam_pg)
        self.base_optimizer = SAM(sam_param_groups, torch.optim.Adam, rho=rho, **kwargs)

        ## other control variables
        self.importance_matrix = importance_matrix if importance_matrix is not None else None
        self.sparsity = sparsity
        self.interval = interval
        self.current_step = 0
        self.prune_n = prune_n
        self.prune_m = prune_m
    
    def update_importance_matrix(self, importance_matrix):
        if len(importance_matrix) != len(self.param_groups):
            raise ValueError("importance_matrix must have the same length as param_groups.")
        self.importance_matrix = importance_matrix

    def final_projection(self):
        for group in self.param_groups:
            if group.get('admm', False):
                weights = group['params']
            final_weights = self.projection(
                weights,
                self.sparsity,
                prune_n=self.prune_n,
                prune_m=self.prune_m,
                importance_matrix=self.importance_matrix
            )
            for w,fw in zip(weights,final_weights):
                w.data.copy_(fw)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self.base_optimizer.first_step(zero_grad)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        
        for group in self.param_groups:
            if group.get('admm', False):
                weights = group['params']
                lmda = group['lmda']
                duals = group['duals']
                splits = group['splits']
                

                if not all([weights, duals, splits]): # Ensure all necessary lists are present
                    print(f"Warning: ADMM group missing weights, duals, or splits. Skipping proximal for this group.")
                    continue
                if not (len(weights) == len(duals) == len(splits)):
                    print(f"Warning: Mismatch in lengths of weights, duals, splits for an ADMM group. Skipping proximal.")
                    continue

                for i in range(len(weights)):
                    if weights[i].grad is None:
                        continue
                    proximal = lmda * (weights[i].detach() - splits[i].detach() + duals[i].detach()) # proximal gradient w-z+u
                    weights[i].grad.add_(proximal)
    
        self.base_optimizer.second_step(zero_grad) # zero_grad is typically False for SAM's second_step

        # Dual update
        if (self.current_step + 1) % self.interval == 0:
            with torch.no_grad():
                for group in self.param_groups:
                    if group.get('admm', False):
                        weights = group['params'] # W_t+1
                        duals = group['duals']   # U_t
                        splits = group['splits'] # Z_t
                        sparsity = self.sparsity

                        if not all([weights, duals, splits]):
                            print(f"Warning: ADMM group missing weights, duals, or splits. Skipping dual/split update for this group.")
                            continue
                        if not (len(weights) == len(duals) == len(splits)):
                            print(f"Warning: Mismatch in lengths of weights, duals, splits for an ADMM group. Skipping dual/split update.")
                            continue

                        z_input = [w.detach() + u.detach() for w, u in zip(weights, duals)] # proj (W_t+1 + U_t)
                        z_new = self.projection(z_input, sparsity, prune_n=self.prune_n, prune_m=self.prune_m, importance_matrix=self.importance_matrix)
                        u_new = [u.detach() + w.detach() - z_n.detach() for u, w, z_n in zip(duals, weights, z_new)] # U_t+1 = U_t + W_t+1 - Z_t+1
                        
                        for i in range(len(duals)): # Iterate using index to ensure correct assignment
                            duals[i].copy_(u_new[i])
                            splits[i].copy_(z_new[i])
        
        self.current_step += 1
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAFE requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

class SAM(torch.optim.Optimizer):
    def __init__(
        self, 
        params,
        base_optimizer: torch.optim.Optimizer,
        rho:int =0.05,
        adaptive:bool =False,
        **kwargs
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        """
        SAM optimizer. Implementation from https://github.com/davda54/sam (SAM)
        Args:
            params (iterable): Parameters to optimize or dicts defining parameter groups.
            base_optimizer (torch.optim.Optimizer): Base optimizer to use.
            rho (float): Perturbation size for SAM.
            adaptive (bool): Whether to use adaptive scaling.
            **kwargs: Additional arguments for the base optimizer.
        """

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            grad_norm = self._grad_norm()
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
