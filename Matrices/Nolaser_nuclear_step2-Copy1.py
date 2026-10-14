import torch
import numpy as np
import math
from Matrices.electronic import calc_NAC_viaHess, calc_NAC_viaHess_H2


Ntraj = 240
nstate = 2
Natoms= 18




#atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_ = 0, 0, 0, 0, 0, 0

def schnet_input(positions_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_,device):
    input_snet = {}
    input_snet['_positions'] = positions_x.to(device)
    input_snet['_atomic_numbers'] = atomic_numbers_.to(device)
    input_snet['_cell_offset'] = cell_offset_.to(device)
    input_snet['_cell'] = cell_.to(device)
    input_snet['_neighbors'] = neighbors_.to(device)
    input_snet['_neighbor_mask'] = neighbor_mask_.to(device)
    input_snet['_atom_mask'] = atom_mask_.to(device)
    
    return input_snet


def Hamiltonian_output(positions_x, model, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, device):
    E_ref = -344.1 #-230.7012
    
    input_snet = schnet_input(positions_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, device)

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


def Hamiltonian_output_nac(positions_x, model, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, device):
    E_ref = -344.1 #-230.7012
    
    input_snet = schnet_input(positions_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, device)

    output_snet = model(input_snet)

    Hess_ = output_snet['hessian'].cpu().detach()
    Energys_ = output_snet['energy'].cpu().detach()
    Forces_ = output_snet['forces'].cpu().detach()
    
    #nstate = Energys_.shape[1]
    #Ntraj = Energys_.shape[0]
    
    H_t = (Energys_.unsqueeze(2)-E_ref)*torch.tile(torch.eye(nstate).unsqueeze(0),[Ntraj,1,1])
    NAC_t = calc_NAC_viaHess(Energys_,Forces_,Hess_)
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

def Dipole_Moment(position_x, polar_vector, model_dp, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_):
    
    position_x = position_x.detach()
    position_x.requires_grad = True
    Dipoles_ = model_dp(schnet_input(position_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_))['dipoles']
    
    DM = torch.zeros([Ntraj,nstate,nstate,1,3])
    DM[:,0,0,:] = Dipoles_[:,0:1]
    DM[:,0,1,:] = Dipoles_[:,1:2]
    DM[:,1,0,:] = Dipoles_[:,1:2]
    DM[:,1,1,:] = Dipoles_[:,2:3]
    
    DM_polar = torch.sum(torch.sum(DM*polar_vector,3),3)
    
    dμdx = torch.zeros([Ntraj,nstate,nstate,Natoms,3])
    
    grad_out = torch.zeros(DM_polar.size())
    grad_out[:,0,0] = torch.ones([Ntraj])
    dμdx[:,0,0,:] = torch.autograd.grad(DM_polar, position_x, grad_outputs=grad_out, retain_graph=True)[0]

    grad_out = torch.zeros(DM_polar.size())
    grad_out[:,0,1] = torch.ones([Ntraj])
    dμdx[:,0,1,:] = torch.autograd.grad(DM_polar, position_x, grad_outputs=grad_out, retain_graph=True)[0]
    dμdx[:,1,0,:] = dμdx[:,0,1,:]

    grad_out = torch.zeros(DM_polar.size())
    grad_out[:,1,1] = torch.ones([Ntraj])
    dμdx[:,1,1,:] = torch.autograd.grad(DM_polar, position_x, grad_outputs=grad_out)[0]
    
    position_x.requires_grad = False
    
    return DM_polar, dμdx

def Dipole_Moment(position_x, model_dp, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_,device):
### calc at cpu device    
    position_x = position_x.detach()
    position_x.requires_grad = True
    sch_in = schnet_input(position_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_,device)
    #sch_in_cpu = {k:v.to('cpu') for k,v in sch_in.items()}
   
    Dipoles_ = model_dp(sch_in)['dipoles']
    
    DM = torch.zeros([Ntraj,nstate,nstate,1,3])
    DM[:,0,0,:] = Dipoles_[:,0:1]
    DM[:,0,1,:] = Dipoles_[:,1:2]
    DM[:,1,0,:] = Dipoles_[:,1:2]
    DM[:,1,1,:] = Dipoles_[:,2:3]
    
    DDipoles_ = torch.zeros([Ntraj,3,3,Natoms,3])
    
    for i in range(3):
        for j in range(3): 
            DDipoles_[:,i,j] = torch.autograd.grad(Dipoles_[:,i,j], position_x, 
                                                   grad_outputs=torch.ones([Ntraj,]).to(device),
                                                   retain_graph=True)[0]
    
    dμdx = torch.zeros([Ntraj,nstate,nstate,3,Natoms,3])
    
    dμdx[:,0,0] = DDipoles_[:,0]
    dμdx[:,0,1] = DDipoles_[:,1]
    dμdx[:,1,0] = DDipoles_[:,1]
    dμdx[:,1,1] = DDipoles_[:,2]
    
    position_x.requires_grad = False
    
    return DM, dμdx


def H_MCH(H):
    
    return H


#def H_Diagonal(Hmch):
    
#    eigvals_ = torch.linalg.eigh(Hmch)[0]
#    eigvecs_ = torch.linalg.eigh(Hmch)[1]
    
#    return eigvals_, eigvecs_

def zlowdin(M):
    
    M = M.to(torch.complex64)
    
    M_hermi = torch.transpose(torch.conj(M),1,2)
    S = torch.matmul(M_hermi,M)
    s, u = torch.linalg.eig(S)
    s_inv_half = (1/torch.sqrt(s).unsqueeze(1))*np.eye(nstate,nstate)
    
    s_inv_half = (s_inv_half.detach()).type(torch.complex64)
    
    u_hermi = torch.transpose(torch.conj(u),1,2)
    
    
    M_ort = torch.matmul(torch.matmul(torch.matmul(u,s_inv_half), u_hermi), M)
    M_lowdin = M_ort/torch.sqrt(torch.sum(torch.abs(M_ort),1)).unsqueeze(1)
    
    return M_lowdin.to(torch.float32)


def Phase_correct_U(eigvecs_t, eigvecs_t_Δt, eigvals_t_Δt):

    diag_degen_diff=1e-9
    
    S = torch.matmul(torch.transpose(torch.conj(eigvecs_t_Δt),1,2), eigvecs_t)
    
    E_val = eigvals_t_Δt.unsqueeze(1)
    
    for i in range(nstate):
        for j in  range(nstate):
            mark_list = np.int32((torch.abs(S)[:,i:i+1,j:j+1])**2 >0.5)*np.int64(torch.abs(E_val[:,:,i:i+1]-E_val[:,:,j:j+1]) > diag_degen_diff)
            
            if i==j:
                continue
            
            else:

                S_gather = torch.zeros(S.size())
                S_gather[:,:,0] = S[:,:,1].detach()
                S_gather[:,:,1] = S[:,:,0].detach()
                S = S_gather*mark_list + S.detach()*(1-mark_list)
                S = S.type(torch.float32)
                
                E_val_gather = torch.zeros(E_val.size())
                E_val_gather[:,:,0] = E_val[:,:,1].detach()
                E_val_gather[:,:,1] = E_val[:,:,0].detach()
                
                E_val = E_val_gather*mark_list + E_val.detach()*(1-mark_list)
                E_val = E_val.type(torch.float32)
            
    Mask = np.int32(torch.abs(E_val[:,:,0:1]-E_val[:,:,0:1])<diag_degen_diff)*np.array([[1,0],[0,0]]) + \
    np.int32(torch.abs(E_val[:,:,0:1]-E_val[:,:,1:2])<diag_degen_diff)*np.array([[0,1],[0,0]]) +\
    np.int32(torch.abs(E_val[:,:,1:2]-E_val[:,:,0:1])<diag_degen_diff)*np.array([[0,0],[1,0]]) +\
    np.int32(torch.abs(E_val[:,:,1:2]-E_val[:,:,1:2])<diag_degen_diff)*np.array([[0,0],[0,1]])
    
    P_project = (S*Mask).type(torch.float32)
    P_project_oth = zlowdin(P_project)

    eigvecs_t_Δt_correct = torch.matmul(eigvecs_t_Δt, P_project_oth)
    
    return eigvecs_t_Δt_correct, P_project_oth



def Interpolation_U(H_old, U_old, H_now):
                 #Hmch_t, eigvecs_t, Hmch_t_Δt

    U_old_dummy = U_old
    
    Nsubstep = 20
    for istep in range(1,Nsubstep+1):
        H = H_old + (H_now-H_old)*istep/Nsubstep ## interpolat overlap H
        
        eigval_H, eigvec_H = H_Diagonal(H)

        U, P_proj = Phase_correct_U(U_old_dummy, eigvec_H,eigval_H)
                
        U_old_dummy = U
        
    return U, eigval_H, P_proj 


def F_Diagonal(F, dμdx, E_laser, eigvecs_correct):

    
    F_diag = torch.zeros([Ntraj,nstate,nstate, Natoms,3])
    F_diag[:,0,0,:] = F[:,0,:]
    F_diag[:,1,1,:] = F[:,1,:]
    
    F_diag = F_diag + torch.sum(dμdx*E_laser,-3)


    ev_expand = torch.tile(eigvecs_correct.unsqueeze(3).unsqueeze(3),(1,1,1,Natoms,3))

    F_diag = torch.matmul(torch.transpose(ev_expand.permute(0,3,4,1,2),3,4),
                          torch.matmul(F_diag.permute(0,3,4,1,2), ev_expand.permute(0,3,4,1,2)))

    F_diag = F_diag.permute(0,3,4,1,2)[:,[0,1],[0,1],:,:]
    
    return F_diag


def propagator(H_isub, K_isub, Δt_small):

    #Hmch 
    Hmch_isub = H_isub #- torch.sum(torch.sum(μ_isub*laser_isub, -1), -1)
    #laser(t_isub, OMega, fwhm, t_c)
    
    expH = torch.linalg.matrix_exp(-(Hmch_isub*1j + K_isub)*Δt_small)
    
    return expH


def Rtotal(H_t, H_t_Δt, NAC_t, NAC_t_Δt, v_t, v_t_Δt, Δt, t):
    
    
    ## NAC
    K_t = torch.sum(torch.sum(NAC_t*(v_t.unsqueeze(1).unsqueeze(1)),3),3)
    K_t_Δt = torch.sum(torch.sum(NAC_t_Δt*(v_t_Δt.unsqueeze(1).unsqueeze(1)),3),3)
    
    Nsubstep = 25
    Δt_small = Δt/Nsubstep
    
    R_total = torch.tile(torch.eye(nstate).unsqueeze(0).type(torch.complex64),[Ntraj,1,1])
    
    for isub in range(1,Nsubstep+1):
        
        H_isub = H_t + (H_t_Δt - H_t)*isub/Nsubstep
        K_isub = K_t + (K_t_Δt - K_t)*isub/Nsubstep
        
        t_isub = t + Δt*isub/Nsubstep
        
        expH = propagator(H_isub, K_isub, Δt_small)
        R_total = torch.matmul(expH, R_total)
    
#    R_total = torch.matmul( torch.matmul(torch.transpose(torch.conj(eigvecs_t_Δt) ,1,2),R_total), eigvecs_t)
    
    return R_total


def update_coefficient(R_total,coefficient):
    
    return torch.matmul(R_total,coefficient)


def Prob_hop(coefficient_t, coefficient_t_Δt, R_total):
    
    Prob_true = torch.zeros([Ntraj,nstate,nstate])
    
    x = torch.transpose(torch.conj(coefficient_t)*coefficient_t, 1,2)
    w = torch.transpose(torch.conj(coefficient_t_Δt)*coefficient_t_Δt, 1,2)
    
    R_diag = torch.sum(R_total*torch.eye(nstate).unsqueeze(0),2).unsqueeze(1)
    
    j = torch.transpose(coefficient_t_Δt,1,2)*torch.conj(R_diag)*torch.transpose(torch.conj(coefficient_t),1,2)
    z = torch.real( torch.transpose(torch.conj(coefficient_t),1,2)*torch.conj(R_total)*coefficient_t_Δt )
    
    y = torch.real(x - j)
    Prob = (1-torch.real(w/x))*torch.real(z/y)
    
    Prob = torch.where(torch.isnan(Prob), torch.full_like(Prob, 0), Prob)
    Prob = torch.where(torch.isinf(Prob), torch.full_like(Prob, 0), Prob)
    #print(Prob)
    
    Prob_true[:,1,0] = Prob[:,1,0]
    Prob_true[:,0,1] = Prob[:,0,1]
    
    return Prob_true


def update_activatestate(activatestate_index, seqtraj_index, Prob, H, v, m, random_num):

    threshold_hop = 0.04  #0.5/27.2114
    
#    random_num = np.random.uniform(0,1,Ntraj).reshape(Ntraj,1,1)
    compara_p_r = torch.tensor(Prob.detach().numpy() > random_num).to(torch.int32)
    hop_index = compara_p_r[seqtraj_index, list(1-np.int32(activatestate_index)), activatestate_index ]
    
    ## determine in gap_thres or out gap_thres hopping
    deter_out = np.int32(np.abs(torch.abs((H[:,1,1] - H[:,0,0])).detach().numpy()) > threshold_hop)
    hop_index_outthreshold = hop_index.detach().numpy()*deter_out
    hop_index_nonadiabatic = hop_index.detach().numpy() - hop_index_outthreshold
    
    ## check the frustrated hopping
    
    activatestate_index_new = list(np.int32(activatestate_index) + ((-1)**np.int32(activatestate_index))*hop_index_nonadiabatic )
    
    
    GAP_hop = H[seqtraj_index,activatestate_index_new,activatestate_index_new] - H[seqtraj_index,activatestate_index,activatestate_index]
    GAP_hop = GAP_hop.detach().numpy()
    
    
    E_kin = torch.sum(torch.sum(0.5*m*v**2,1),1).detach().numpy() #,
    
    #E_kin > GAP_hop, 
    
    deter_normalhop = np.int32(E_kin > GAP_hop)
    
    hop_index_normalhop = hop_index_nonadiabatic*deter_normalhop
    hop_index_frusthop = hop_index_nonadiabatic - hop_index_normalhop
    
    
    f_factor = np.sqrt(np.abs((E_kin - GAP_hop)/E_kin))*hop_index_normalhop + (1-hop_index_normalhop)
    f_factor = torch.tensor(f_factor).type(torch.float32).unsqueeze(1).unsqueeze(1)
    
    hop_index_True = hop_index_normalhop
    
    
    activatestate_index_next = list(np.int32(activatestate_index) + ((-1)**np.int32(activatestate_index))*hop_index_True)
    
    return activatestate_index_next, f_factor


def EDC_decoherence(coefficients, activatestate_index, seqtraj_index, H, v, m, Δt):

    c_α = coefficients[seqtraj_index,list(1-np.int32(activatestate_index)),:]
    c_β = coefficients[seqtraj_index,activatestate_index,:]
    
    E_kin = torch.sum(torch.sum(0.5*m*v**2,1),1)
    
#    gap = torch.abs(eigvals[:,1] - eigvals[:,0])
    gap = torch.abs(H[:,1,1] - H[:,0,0])
    
    scale_α = torch.exp(-0.5*Δt*gap*(1/(1 + 0.1/E_kin)))
    c_α_p = scale_α.unsqueeze(1)*c_α
    
    scale_β = torch.sqrt(1 - torch.abs(c_α_p)**2)/torch.abs(c_β)
    c_β_p = scale_β*c_β
    
    coefficients[seqtraj_index,list(1-np.int32(activatestate_index)),:] = c_α_p
    coefficients[seqtraj_index,activatestate_index,:] = c_β_p
    
    return coefficients
