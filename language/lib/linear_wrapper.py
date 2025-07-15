
## Pruner wrapper implementation for SparseGPT, Wanda, ALPS.
## Each implementation is adopted from the original paper.

import math
import time
from typing import Any
from abc import ABCMeta, abstractmethod

from absl import logging
import numpy as np
import torch
import torch.nn as nn
import transformers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# Linear Wrapper Base class.
# -----------------------------------------------------------------------------

class LinearPrunerWrapperBase:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.nsamples = 0
        self.init_info()
        assert hasattr(self, 'info_name'), \
            "info_name attribute must be defined in the subclass"
        
    @property
    @abstractmethod
    def PRUNER_NAME(self):
        pass
    
    def update_nsamples(self, inp):
        """update n_samples and info by batch size of new inp"""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        curr_batch_size = inp.shape[0]

        if getattr(self, 'renormalize_info', True):
            info = getattr(self, self.info_name)
            renorm_info = info * self.nsamples / (self.nsamples+curr_batch_size)
            setattr(self, self.info_name, renorm_info)
        
        self.nsamples += curr_batch_size
    
    def preprocess_input(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            logging.info(inp.shape)
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        return inp


    def add_batch(self, inp, out):
        self.update_nsamples(inp)
        inp = self.preprocess_input(inp)
        new_info = getattr(self, self.info_name)+self.compute_per_batch_info(inp)
        setattr(self, self.info_name, new_info)
    
        
    @abstractmethod
    def init_info(self):
        """
        Define attribute and initializes the Hessian or Activation variable with the layer's weight information.
        It must also define the `info_name` attribute, which is used to store the information.
        This method should be overridden in subclasses to provide specific initialization logic.
        """
        pass
    
    @abstractmethod
    def compute_per_batch_info(self, inp):
        pass
    
    @abstractmethod
    def prune_linear(self, sparsity, prune_n=0, prune_m=0):
        pass 



# Linear pruners (Wanda, SparseGPT, Alps).
# -----------------------------------------------------------------------------

class WandaWrapper(LinearPrunerWrapperBase):
    """
    Wraps linear layer to prune using the Wanda (\|w\|\|x\|_2) method.
    Original code: https://github.com/locuslab/wanda
    """
    PRUNER_NAME = "Wanda"

    def init_info(self):
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.info_name = "scaler_row" 

    def compute_per_batch_info(self, inp):
        inp = inp.float()
        return torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

    def prune_linear(
        self, sparsity, prune_n=0, prune_m=0, 
    ):
        W = self.layer.weight.data.clone()
        # WANDA Metric: |W| * sqrt(Activation_Scale for each input neuron, i.e., row of W)
        # scaler_row corresponds to ||X_j||_2 / sqrt(num_samples) for each input feature j
        # So, W_metric_ij = |W_ij| * ||X_j||_2
        activation_scales = torch.sqrt(self.scaler_row.reshape((1, -1))).to(self.dev) 
        W_metric = torch.abs(W) * activation_scales

        W_mask = torch.zeros_like(W, dtype=torch.bool) 
        if prune_n != 0 and prune_m != 0: # N:M structured sparsity
            for col_chunk_idx in range(W_metric.shape[1] // prune_m):
                start_col = col_chunk_idx * prune_m
                end_col = start_col + prune_m
                tmp_metric_chunk = W_metric[:, start_col:end_col]
                _, topk_indices = torch.topk(tmp_metric_chunk, prune_n, dim=1, largest=False)
                W_mask[:, start_col:end_col].scatter_(1, topk_indices, True)
        else: # Unstructured sparsity
            num_cols_to_prune_per_row = int(W_metric.shape[1] * sparsity)
            if num_cols_to_prune_per_row > 0:
                _, indices_to_prune = torch.topk(W_metric, num_cols_to_prune_per_row, dim=1, largest=False)
                W_mask.scatter_(1, indices_to_prune, True)
        
        W[W_mask] = 0  # Apply pruning mask
            
        self.layer.weight.data = W

    def free(self):
        self.scaler_row = None
        torch.cuda.empty_cache()



class SparseGPTWrapper(LinearPrunerWrapperBase):
    """
    Wraps linear layer to prune using the SparseGPT method.
    Original code: https://github.com/IST-DASLab/sparsegpt
    """
    PRUNER_NAME = "SparseGPT"

    def init_info(self):
        self.H = torch.zeros((self.columns, self.columns), device=self.dev).float()
        self.info_name = "H"  # Name of the attribute to store the Hessian information

    def compute_per_batch_info(self, inp):
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        return inp.matmul(inp.t())

    def prune_linear(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        # distance = torch.sum((W-self.layer.weight.data)**2)
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()



class ALPSWrapper(LinearPrunerWrapperBase):
    """ 
    Wraps linear layer to prune using the ALPS method.
    Original code: https://github.com/mazumder-lab/ALPS
    """
    PRUNER_NAME = "ALPS"
    renormalize_info = False  # ALPS does not renormalize the Hessian information when adding new batches

    def init_info(self):
        self.H = torch.zeros((self.columns, self.columns), device=self.dev).float()
        self.info_name = "H"  # Name of the attribute to store the Hessian information
    
    def compute_per_batch_info(self, inp):
        inp = inp.float()
        return inp.matmul(inp.t())

    def prune_linear(
        self, sparsity, prune_n=0, prune_m=0, rho=0.1, max_iter = 300, update_iter = 3, switch_iter = 30, fix_supp=False
    ):
        
        # get dense weight
        W = self.layer.weight.data.clone()
        W = W.float()
        W = W.to('cuda:0')
        self.H = self.H.cpu()
        
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
            
        # ridge term
        damp1 = 0.01 * torch.mean(torch.diag(self.H)).item()
        diag = torch.arange(self.H.shape[0], device=self.H.device)
        self.H[diag,diag] += damp1
        
        # normalization 
        X_norm = torch.diag(self.H).sqrt() + 1e-8
        self.H = self.H / X_norm
        self.H = (self.H.T / X_norm).T    
        
        self.YtX = torch.zeros_like(W)
        self.YtX = torch.matmul(W.cpu() * X_norm,self.H).to(self.dev)

        admm_st = time.time()

        # initialization
        XTX_inv = torch.zeros_like(self.H).float().to('cuda:0')
        B = (W * X_norm.to(self.dev)).t().clone()
        W = None
        B_orig = B.cpu().clone()
        V = torch.zeros_like(B)
        D = torch.zeros_like(B)
        D_suppp = torch.zeros_like(B)
        D_supp = torch.zeros_like(B)

        
        totp, num_cout = B.shape
        L, Q = torch.linalg.eigh(self.H.double())
        XTX_inv = (Q @ ((1/(L+(rho))) * Q).T).float().to(self.dev)
        
        init_rho = False
        fix_supp = False
        D_fix = torch.zeros_like(D)
        
        Res0 = self.YtX.T.cpu()
        Res0 = torch.sum(B_orig.cpu() * Res0)
        Res0 = torch.sum(Res0)

        params = B.shape[0]*B.shape[1]
        k_spar = int(np.round((1-sparsity)*params))
    
        
        if prune_n == 0:
            D = B.clone().reshape(-1)
            _, loss_idx = torch.topk(-D**2,totp * num_cout - k_spar)
            D[loss_idx] = 0    
            D_suppp = (D == 0).to(torch.float)
            D = D.reshape(totp, num_cout)
        else:
            new_dim = totp * num_cout / prune_m
            new_dim = int(new_dim)
            k_spar = totp * num_cout * prune_n/prune_m
            
        
            D = B.clone().t().reshape((new_dim, prune_m))
            _, loss_idx = torch.topk(-D**2,prune_m - prune_n, dim = 1)
            D = D.scatter(src=torch.zeros((new_dim,prune_m-prune_n)).to('cuda:0'),dim=1,index=loss_idx)   
            D_suppp = (D == 0).to(torch.float)
            D = D.reshape(num_cout, totp).t()
    
        D_init = D.clone()
        errorp = 1
        for i_admm in range(max_iter):
      

            B = XTX_inv @ (self.YtX.T-V+rho*D)

            if fix_supp:
                D = ((V + rho * B) / rho) * D_fix
            elif prune_n == 0:
                D = ((V + rho * B) / rho).reshape(-1)
                _, loss_idx = torch.topk(-D**2,totp * num_cout - k_spar)
                D[loss_idx] = 0    
                D = D.reshape(totp, num_cout)   
            else:
                D = ((V + rho * B) / rho).t().reshape((new_dim, prune_m))
                _, loss_idx = torch.topk(-D**2,prune_m - prune_n, dim = 1)
                D = D.scatter(src=torch.zeros((new_dim,prune_m-prune_n)).to('cuda:0'),dim=1,index=loss_idx) 
                D_supp = (D == 0).to(torch.float)  
                D = D.reshape(num_cout, totp).t()  

            V = V + rho * (B - D)
            distance = torch.sqrt(torch.sum((B-D)**2))
            logging.info("iter {}, distance {}".format(i_admm, distance))
            if (i_admm+1) % update_iter == 0:

         
                if prune_n == 0:
                    D_supp = (D.reshape(-1) == 0).to(torch.float)
                supp_change = torch.sum((D_supp-D_suppp)**2)
                
                if not fix_supp:
                    if supp_change / k_spar > 0.1:
                        init_rho = True
                        rho *= 1.3
                    elif supp_change / k_spar > 0.005:
                        init_rho = True
                        rho *= 1.2
                    elif supp_change > 0.5:
                        if init_rho:
                            rho *= 1.1
                        else:
                            rho /= 5
                            B = B_orig.clone().to(self.dev)
                            D = D_init.clone().to(self.dev)
                            V = torch.zeros_like(B).to(self.dev)     
                    else:
                        if init_rho:
                            break
                        else:
                            rho /= 5
                
                D_suppp = (D_supp).clone()
                if rho > 1e6:
                    rho = 1e6
               
                XTX_inv = (Q @ ((1/(L+(rho))) * Q).T).float().to(self.dev)
                
                if prune_n == 0:
                    Btest = B.reshape(-1)
                    _, loss_idx = torch.topk(-Btest**2,totp * num_cout - k_spar)
                    Btest[loss_idx] = 0    
                    Btest = Btest.reshape(totp, num_cout)
                else:
                    Btest = B.t().reshape((new_dim, prune_m))
                    _, loss_idx = torch.topk(-Btest**2,prune_m - prune_n, dim = 1)
                    Btest = Btest.scatter(src=torch.zeros((new_dim,prune_m-prune_n)).to('cuda:0'),dim=1,index=loss_idx)  
                    Btest = Btest.reshape(num_cout, totp).t()
            
                Resc = torch.matmul(self.H.to(self.dev),Btest) - self.YtX.T
                Resc = torch.diag(torch.matmul((Btest-B_orig.to(self.dev)).t(), Resc))

        
                errorc = torch.sum(Resc).to("cpu")/Res0
                errorc = errorc.item()
                logging.info("iter {}, error {} support change {}, rho {}".format(i_admm, errorc / errorp, supp_change / k_spar, rho))
                
                
                if i_admm >= switch_iter and supp_change / k_spar < 0.0003:
                    break

        if prune_n == 0:
            B = B.reshape(-1)
            _, loss_idx = torch.topk(-B**2,totp * num_cout - k_spar)
            B[loss_idx] = 0    
            B = B.reshape(totp, num_cout)
        else:
            B = B.t().reshape((new_dim, prune_m))
            _, loss_idx = torch.topk(-B**2,prune_m - prune_n, dim = 1)
            B = B.scatter(src=torch.zeros((new_dim,prune_m-prune_n)).to('cuda:0'),dim=1,index=loss_idx)  
            B = B.reshape(num_cout, totp).t()

        V = None
        D = None

        Res = torch.matmul(self.H,B.cpu() ) - self.YtX.T.cpu()
        Res = torch.diag(torch.matmul((B.cpu()  -B_orig).t(), Res))
        
        error = torch.sum(Res)/Res0
        error = error.item()

        #logging.info("Before backsolve, error is {}".format(error))
        admm_time = time.time() - admm_st
        
        back_st = time.time()
        B = self.cg_batch( (self.H).to(self.dev), self.YtX.T, 
                    (B != 0).to(torch.float), M_bmm=None, X0=B, rtol=1e-4, atol=0., maxiter=10, verbose=False)
            
        back_time = time.time() - back_st
        
        
        Res = torch.matmul(self.H,B.cpu() ) - self.YtX.T.cpu()
        Res = torch.diag(torch.matmul((B.cpu()  -B_orig).t(), Res))
        
        error = torch.sum(Res)/Res0
        error = error.item()
        
        torch.cuda.synchronize()

        if isinstance(self.layer, transformers.Conv1D):
            self.layer.weight.data = (B.t() / X_norm.to(self.dev)).t().reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        else:
            self.layer.weight.data = (B.t() / X_norm.to(self.dev)).reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


        return error
    
    # A modified version of https://github.com/sbarratt/torch_cg
    def cg_batch(self, A, B, A_supp, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
        error_list = np.zeros((maxiter,))
        n, m = B.shape

        if M_bmm is None:
            M_bmm = lambda x: x
        if X0 is None:
            X0 = M_bmm(B)
        if maxiter is None:
            maxiter = 5 * n

        assert B.shape == (n, m)
        assert X0.shape == (n, m)
        assert rtol > 0 or atol > 0
        assert isinstance(maxiter, int)

        X_k = X0
    
        R_k = B - A @ X_k
        R_k = R_k * A_supp
    
        Z_k = M_bmm(R_k)

        P_k = torch.zeros_like(Z_k)

        P_k1 = P_k
        R_k1 = R_k
        R_k2 = R_k
        X_k1 = X0
        Z_k1 = Z_k
        Z_k2 = Z_k

        B_norm = torch.norm(B, dim=1)
        stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

        if verbose:
            logging.info("%03s | %010s %06s" % ("it", "dist", "it/s"))

        optimal = False
        start = time.perf_counter()
        for k in range(1, maxiter + 1):
            start_iter = time.perf_counter()
            Z_k = M_bmm(R_k)

            if k == 1:
                P_k = Z_k
                R_k1 = R_k
                X_k1 = X_k
                Z_k1 = Z_k
            else:
                R_k2 = R_k1
                Z_k2 = Z_k1
                P_k1 = P_k
                R_k1 = R_k
                Z_k1 = Z_k
                X_k1 = X_k
                denominator = (R_k2 * Z_k2).sum(0)
                denominator[denominator == 0] = 1e-8
                beta = (R_k1 * Z_k1).sum(0) / denominator
                P_k = Z_k1 + beta.unsqueeze(0) * P_k1

            denominator = (P_k * (A@P_k)).sum(0)
            denominator[denominator == 0] = 1e-8
            alpha = (R_k1 * Z_k1).sum(0) / denominator
            X_k = X_k1 + alpha.unsqueeze(0) * P_k
            R_k = R_k1 - alpha.unsqueeze(0) * (A@P_k)
            R_k = R_k * A_supp
            end_iter = time.perf_counter()

            residual_norm = torch.norm(A@X_k - B, dim=1)

            if verbose:
                logging.info("%03d | %8.4e" %
                      (k, torch.max(residual_norm/B_norm)))

            if (residual_norm <= stopping_matrix).all():
                optimal = True
                break


        end = time.perf_counter()

        if verbose:
            if optimal:
                logging.info("Terminated in %d steps (optimal). Took %.3f ms." %
                      (k, (end - start) * 1000))
            else:
                logging.info("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                      (k, (end - start) * 1000))


        info = {
            "niter": k,
            "optimal": optimal
        }

        return X_k


    def XtX_inv(self, rho):

        if self.QQtlow is None:
            XtXInv = torch.zeros_like(self.H).cpu().double()
        else:
            XtXInv = self.QQtlow /rho
        
        XtXInv += self.Q_high @ torch.diag(torch.reciprocal(self.L_high + torch.ones_like(self.L_high).cpu().double()*rho)) @ self.Q_high.T
        XtXInv = XtXInv.float().to('cuda:0')
        return XtXInv
        


    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        self.X = None
        self.Y = None
        self.XXt = None
        self.YXt = None
        self.YtX = None
        self.H = None
        torch.cuda.empty_cache()





class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp
        inp = inp.float()
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
