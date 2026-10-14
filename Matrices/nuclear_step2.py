import torch
import numpy as np
import math
from Matrices.electronic import calc_NAC_viaHess, calc_NAC_viaHess_H2





#atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_ = 0, 0, 0, 0, 0, 0

def schnet_input(positions_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_):
    input_snet = {}
    input_snet['_positions'] = positions_x
    input_snet['_atomic_numbers'] = atomic_numbers_
    input_snet['_cell_offset'] = cell_offset_
    input_snet['_cell'] = cell_
    input_snet['_neighbors'] = neighbors_
    input_snet['_neighbor_mask'] = neighbor_mask_
    input_snet['_atom_mask'] = atom_mask_
    
    return input_snet


def Hamiltonian_output(positions_x, model, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_):
    E_ref = -230.7012
    
    input_snet = schnet_input(positions_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_)

    output_snet = model(input_snet)

    #Hess_ = output_snet['hessian'].cpu().detach()
    Energys_ = output_snet['energy'].cpu().detach()
    Forces_ = output_snet['forces'].cpu().detach()
    
    #nstate = Energys_.shape[1]
    #Ntraj = Energys_.shape[0]
    H_t = (Energys_.unsqueeze(2)-E_ref)*torch.tile(torch.eye(nstate).unsqueeze(0),[Ntraj,1,1])
    #NAC_t = calc_NAC_viaHess(Energys_,Forces_,Hess_)
    F_t = Forces_   #[seqtraj_index,activatestate_index,:]
    
    return H_t, F_t #, NAC_t


def Hamiltonian_output_nac(positions_x, model, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_):
    E_ref = -230.7012
    
    input_snet = schnet_input(positions_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_)

    output_snet = model(input_snet)

    Hess_ = output_snet['hessian'].cpu().detach()
    Energys_ = output_snet['energy'].cpu().detach()
    Forces_ = output_snet['forces'].cpu().detach()
    
    nstate = Energys_.shape[1]
    Ntraj = Energys_.shape[0]
    
    H_t = (Energys_.unsqueeze(2)-E_ref)*torch.tile(torch.eye(nstate).unsqueeze(0),[Ntraj,1,1])
    NAC_t = calc_NAC_viaHess_H2(Energys_,Forces_,Hess_)
    F_t = Forces_
    
    return H_t, F_t , NAC_t


def vverlet_X(x_t, v_t, F_t, m, Δt):
    a_t = F_t/m
    x_t_Δt = x_t + v_t*Δt + 0.5*a_t*Δt**2
    
    return x_t_Δt

def vverlet_V(v_t, F_t, F_t_Δt, m, Δt):
    a_t = F_t/m
    a_t_Δt = F_t_Δt/m
    v_t_Δt = v_t + 0.5*(a_t + a_t_Δt)*Δt
    
    return v_t_Δt

