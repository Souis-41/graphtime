from __future__ import print_function
import torch
import itertools as _itrt

def Ising_tmatrix(nspins, alpha=0.1, gamma=0.95, ratematrix=False, device=torch.device('cpu')):
    """
        Implements Glaubers master equation variant of the (1D) Ising model with periodic boundary conditions

        (J Math Phys 4 294 (1963); doi: 10.1063/1.1703954)
        nspins: number of spins in model (Note: returns 2^nspins times 2^nspins matrix)
        alpha: basal spin flip-rate, defines time-scale (=0.1)
        gamma: gamma is equal to tanh(\beta 2J) where J is the spin-spin coupling constant in a corresponding Ising model, and \beta is the inverse temperature. 
        ratematrix: return rate matrix as well
    """
    ## original s [(2 ** nspins) X (2 ** nspins)]
    s_th = torch.Tensor(list(_itrt.product([-1, 1], repeat = nspins)), device=device)
    ## original c, stacked for all i [(2 ** nspins) X (2 ** nspins) X nspins]
    c_th = s_th.repeat(s_th.shape[0], 1, 1)

    ## [(2 ** nspins) X (2 ** nspins) X nspins] used to count the difference
    flip_mask_th = (s_th.unsqueeze(1).repeat(1, s_th.shape[0], 1) == c_th)
    ## so that we could filter out len(flipped != 1)
    n_flipped_mask = ((~flip_mask_th).sum(-1) == 1).to(torch.int)

    ## ~ => reverse mask, so that different elements are bared
    ## cumsum => so that all elements after the first different element is non-zero
    ## == 0 => so that we can count the index for first different element
    f_th = ((~ flip_mask_th).to(torch.int).cumsum(-1) == 0).sum(-1)
    ## make diag invalid, make len(flipped != 1) invalid
    valid_mask = (1 - torch.eye(2**nspins)) * n_flipped_mask

    ## there should be an extra layer to contain (len == 3) issue
    ## TODO: Souis: what if nspins == 1?
    s_th_cat = torch.cat(
        [s_th.unsqueeze(1).repeat(1, s_th.shape[0], 1), torch.zeros_like(c_th)[:, :, :1]],
        dim=-1)

    # s[f], s[f+1], s[f-1]
    s_f = torch.gather(s_th_cat, -1, f_th.unsqueeze(-1))
    s_fp1 = torch.gather(s_th_cat, -1, ((f_th + 1) % nspins).unsqueeze(-1))
    s_fm1 = torch.gather(s_th_cat, -1, ((f_th - 1) % nspins).unsqueeze(-1))

    # w, clear diagonal, clear invalid
    W_th = 0.5*alpha*(1.-0.5*gamma*s_f*(s_fm1 + s_fp1))[:, :, 0]
    W_th = W_th * valid_mask

    # fill diagonal
    W_th = W_th + torch.eye(2**nspins, device=device) * -W_th.sum(1)
    ## this is only valid with torch >= 1.7.1
    ## TODO: Souis
    ## torch.matrix_exp, while intends to have same behavior as scipy.linalg.expm, yields different results
    ## https://github.com/pytorch/pytorch/issues/48299
    T_th = torch.matrix_exp(W_th)

    if ratematrix:
        return T, W
    else:
        return T

def Ising_to_discrete_state_th(X):
    """
        Maps a trajectory of spin-states to a corresponding trajectory of unique states. 
            Useful when estimating models with global discretization fx. MSMs. 

        X : list of torch Tensors( trajectories of spin states (T, M) ) where T is the number of time-steps and M is the number of binary spins
    
            returns:
            dts : list of lists ( discrete state trajectories)
    """
    dts = []
    for x in X:
        x = torch.clip(x, min=0)
        dts.append ([int(''.join(map(str, f)), 2) for f in x])
    return dts

def all_Ising_states_th(nspins, device=torch.device('cpu')):
    return torch.Tensor(list(_itrt.product([-1, 1], repeat = nspins)), device=device)

