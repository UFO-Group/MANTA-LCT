import torch
from torch import nn
import numpy as np
import math
from Matrices.electronic import calc_NAC_viaHess, calc_NAC_viaHess_H2
from scipy.interpolate import interp2d, interpn

scale_L=1.8897259885789233 #coord : 1 Astrom = 1.8897259885789233 bohr

Ntraj = 500
nstate = 2
Natoms= 18

## for grid of Q_cv
Ndimb = 25
Ndimd = 29
xyzcoord_Bohr = np.load('./Qtenser/caspt2_struct/3dscan_xyz/eckart_remove_Rot/xyzbohr_3dscan_eckart.npy')

bond = np.linspace(1.1, 3.5, Ndimb)*scale_L     ## units: bohr
dihed = np.linspace(-70,70,Ndimd)*np.pi/180     ## units: deg.

# x1: bond (C--N1) : [0,16]
# x2: bond (C--N2) : [1,17]
# x3: dihedral (C--N1--N2--C) : [0, 16, 17, 1]
x2, x1, x3 = np.meshgrid(bond, bond, dihed)





class XQfunct(nn.Module):
    def __init__(self):
        super(XQfunct, self).__init__()
        
        self.layerkan = KAN([2, 5, 5, Natoms*3], base_activation=nn.Identity) 

    
    def forward(self, inputs_): # inputs_ is Q  is cvs[ Ntrajs x 2]
        
    
        z = self.layerkan(inputs_)
        
        return z ## xyzcoord [Ntrajs x 3*Natoms]
    
    

def frame_vector(X, m):  ## X is struct from traj (units: bohr)
    
    Xc = ( torch.sum(X*m, 1)/ torch.sum(m[:,:,0]) ).unsqueeze(1)
    Xalig = X - Xc

    ## two group for up half (2 N-atoms) and down half (4 C-atoms)
    list_upmol = np.array([16,17])
    list_downmol = np.array([2,5,10,13])
    
    ## two group for forward half (2 C-atoms) and backward half (z C-atoms)
    list_forwmol = np.array([10,13])
    list_backmol = np.array([2,5])

    ## calc the geometrical center for each segment

    upmol_c = torch.mean(Xalig[:,list_upmol],1).unsqueeze(1)
    downmol_c = torch.mean(Xalig[:,list_downmol],1).unsqueeze(1)
    
    forwmol_c = torch.mean(Xalig[:,list_forwmol],1).unsqueeze(1)
    backmol_c = torch.mean(Xalig[:,list_backmol],1).unsqueeze(1)

    ## calc the frame
    
    vec_z = (upmol_c - downmol_c)/torch.sqrt(torch.sum(torch.square(upmol_c - downmol_c), -1)).unsqueeze(-1)
    vec_y = (forwmol_c - backmol_c)/torch.sqrt(torch.sum(torch.square(forwmol_c - backmol_c), -1)).unsqueeze(-1)
    vec_x = torch.cross(vec_z,vec_y)
    
    vec_x_norm = vec_x/torch.sqrt(torch.sum(torch.square(vec_x),-1)).unsqueeze(-1)
    vec_z_norm = vec_z
    vec_y_norm = torch.cross(vec_x_norm, vec_z_norm)

    return vec_x_norm, vec_y_norm, vec_z_norm, Xc
    
    
def Derivative_XQ(Qsv, Rtot, Xc, XQfunct):
    
    Qsv.requires_grad = True
    Xref = torch.matmul( XQfunct(Qsv).reshape(Ntraj,Natoms,3), Rtot) + Xc
    Xbohr = Xref.reshape(Ntraj, Natoms*3)
    
    dXdQ = torch.zeros([Ntraj, Natoms*3, 2])

    for idx in range(Natoms*3):
        dXdQ_idx = torch.autograd.grad(Xbohr[:,idx], Qsv, grad_outputs=torch.ones([Ntraj]), retain_graph=True)[0]    
        dXdQ[:,idx] = dXdQ_idx
        
        
    return dXdQ, Xref


def num_gradients(f, Q):
    Q_alt = Q[np.array([1,0])]
    h = 1e-4
    grads = np.append((f(Q_alt[0]+h, Q_alt[1]) - f(Q_alt[0]-h, Q_alt[1]))/(2*h),
                      (f(Q_alt[0], Q_alt[1]+h) - f(Q_alt[0], Q_alt[1]-h))/(2*h))
    grads_alt = grads[np.array([1,0])]
    return grads_alt
    
def calcdihCNC(x_numpy):

    vec01 = x_numpy[:,16,:] - x_numpy[:,0,:]
    vec01 = vec01/np.expand_dims(np.sqrt(np.sum(np.square(vec01),-1)),-1)
    
    vec12 = x_numpy[:,17,:] - x_numpy[:,16,:]
    vec12 = vec12/np.expand_dims(np.sqrt(np.sum(np.square(vec12),-1)),-1)
    
    vec23 = x_numpy[:,1,:] - x_numpy[:,17,:]
    vec23 = vec23/np.expand_dims(np.sqrt(np.sum(np.square(vec23),-1)),-1)
    
    n1 = np.cross(vec01,vec12,-1)/np.expand_dims( np.sqrt(np.sum(np.square(np.cross(vec01,vec12,-1)),-1)), -1 )
    n2 = np.cross(vec12,vec23,-1)/np.expand_dims( np.sqrt(np.sum(np.square(np.cross(vec12,vec23,-1)),-1)), -1 )
    
    cos_ = np.sum(n1*n2,-1)
    
    sin_ = np.sum(np.cross(n1,n2,-1)*vec12,-1)
    
    theta = calctheta(cos_, sin_)
    
    ## remove the edg of dihedral at 68.5 deg.
    list_exedg = np.where(np.abs(theta) > 67.5*np.pi/180)[0]
    modify_theta = np.sign(theta[list_exedg])*(67.5*np.pi/180)
    theta[list_exedg] = modify_theta
    
    return theta  


def calctheta(cos_dih, sin_dih): ### give cos_or_sin_name // return the true angle value (rad.)
    diheddata = np.zeros_like(cos_dih)
    
    
    La = np.where(cos_dih > np.sqrt(2)/2)
    Lb = np.where(sin_dih > np.sqrt(2)/2)
    Lc = np.where(cos_dih < -np.sqrt(2)/2)
    Ld = np.where(sin_dih < -np.sqrt(2)/2)
    
    
    diheddata[La[0]] =         np.arcsin(sin_dih[La[0]]) #*180/np.pi
    diheddata[Lb[0]] =         np.arccos(cos_dih[Lb[0]]) #*180/np.pi
    diheddata[Lc[0]] = (np.pi-np.arcsin(sin_dih[Lc[0]])) #*180/np.pi
    diheddata[Ld[0]] =        -np.arccos(cos_dih[Ld[0]]) #*180/np.pi
    
    return diheddata


def num_gradients(zvalue, Q):
    
    h = 1e-4
    
    dz = interpn( (bond,bond,dihed), zvalue, Q+np.array([[h,0.,0.]]),  method = 'linear')-interpn( (bond,bond,dihed), zvalue, Q-np.array([[h,0.,0.]]),  method = 'linear')
    dzdq1 = dz/(2*h)
    
    dz = interpn( (bond,bond,dihed), zvalue, Q+np.array([[0.,h,0.]]),  method = 'linear')-interpn( (bond,bond,dihed), zvalue, Q-np.array([[0.,h,0.]]),  method = 'linear')
    dzdq2 = dz/(2*h)
    
    grads_num = np.concatenate((np.expand_dims(dzdq1,-1),
                                np.expand_dims(dzdq2,-1)),axis=-1)
    
    return grads_num


def Jacob_DXDQ(xyzcoord_Bohr, Q):

    dXdQ = np.zeros([Ntraj,Natoms,3,2])
    for iatom in range(Natoms):
        for ixyz in range(3):
        
            dXdQ[:,iatom, ixyz, :] = num_gradients(xyzcoord_Bohr[:,:,:,iatom, ixyz], Q)  ## shape: [Ntraj x 2]
            
            
    dXdQ = torch.tensor(dXdQ.reshape(Ntraj, Natoms*3, 2)).type(torch.float32)
    
    return dXdQ
    

def schnet_input(positions_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device):
    
    ### remove the mass center of positions_X @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    xnp = positions_x.cpu().detach().numpy()  ## nograds positions (ndarry)
    massc = torch.sum( torch.tensor(xnp).type(torch.float32)*m,1).unsqueeze(1)/torch.sum(m[:,:,0])
    positions_xc = positions_x - massc
    ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    input_snet = {}
    input_snet['_positions'] = positions_xc.to(device)
    input_snet['_atomic_numbers'] = atomic_numbers_.to(device)
    input_snet['_cell_offset'] = cell_offset_.to(device)
    input_snet['_cell'] = cell_.to(device)
    input_snet['_neighbors'] = neighbors_.to(device)
    input_snet['_neighbor_mask'] = neighbor_mask_.to(device)
    input_snet['_atom_mask'] = atom_mask_.to(device)
    
    return input_snet


def Hamiltonian_output(positions_x, model, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device):
    E_ref = -344.1 #-230.7012
    
    input_snet = schnet_input(positions_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device)

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


def Hamiltonian_output_nac(positions_x, model, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device):
    E_ref = -344.1 #-230.7012
    
    input_snet = schnet_input(positions_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device)

    output_snet = model(input_snet)

    Hess_ = output_snet['hessian'].cpu().detach()
    Energys_ = output_snet['energy'].cpu().detach()
    Forces_ = output_snet['forces'].cpu().detach()
    
    #nstate = Energys_.shape[1]
    #Ntraj = Energys_.shape[0]
    
    H_t = (Energys_.unsqueeze(2)-E_ref)*torch.tile(torch.eye(nstate).unsqueeze(0),[Ntraj,1,1])
    NAC_t = calc_NAC_viaHess(Energys_,Forces_,Hess_)
    F_t = Forces_
    
    return H_t, F_t , NAC_t, Hess_


def vverlet_X(x_t, v_t, F_t, m, Δt):
    a_t = F_t/m
    x_t_Δt = x_t + v_t*Δt + 0.5*a_t*Δt**2
    
    return x_t_Δt

def vverlet_V(v_t, F_t, F_t_Δt, m, Δt):
    a_t = F_t/m
    a_t_Δt = F_t_Δt/m
    v_t_Δt = v_t + 0.5*(a_t + a_t_Δt)*Δt
    
    return v_t_Δt



def Dipole_Moment(position_x, model_dp, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device):
### calc at cpu device    
    position_x = position_x.detach()
    position_x.requires_grad = True
    sch_in = schnet_input(position_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device)
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

#def Gas_envelope(t,fwhm,t_c):
#    Beta = 4*math.log(2)/(fwhm**2)
#    return math.exp(-Beta*(t-t_c)**2)


#def laser(t,OMega,fwhm,t_c):
#    return 0.05*math.cos(OMega*(t-t_c))*Gas_envelope(t,fwhm,t_c) #+ 0.002*math.cos(0.0260*(t-70*41.34110546116003))*Gas_envelope(t,20*41.34110546116003,70*41.34110546116003) +  0.002*math.cos(0.0260*(t-110*41.34110546116003))*Gas_envelope(t,20*41.34110546116003,110*41.34110546116003) + 0.002*math.cos(0.0260*(t-150*41.34110546116003))*Gas_envelope(t,20*41.34110546116003,150*41.34110546116003)

def actfunct(r):  ## r unit: bohr
    a = 1.
    b = 2.1*1.8897259885789233
    return -0.5*torch.tanh(r*a-b*a)+0.5

def sctfunct(r):
#    n=10
#    n_=50
#    r0=2.0*1.8897259885789233
    
    n=9
    n_=16
    r0=2.1*1.8897259885789233
    
    return (1-(r/r0)**n)/(1-(r/r0)**n_)


def fmulti_funct(rx,ry):
    scale_L = 1.8897259885789233
#    a = scale_L
#    r0 = 2.1*scale_L
    a = 50/scale_L
    r0 = 2.8*scale_L
    
    Z = torch.tanh(-a*(torch.sqrt( rx**2 + ry**2 ) - r0))
    
    return torch.maximum(Z,torch.zeros_like(Z))


def H_MCH(H, μ, E_laser):
    
    return H - torch.sum(torch.sum(μ*E_laser,-1),-1)


def H_Diagonal(Hmch):
    
    eigvals_ = torch.linalg.eigh(Hmch)[0]
    eigvecs_ = torch.linalg.eigh(Hmch)[1]
    
    return eigvals_, eigvecs_

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


def propagator(H_isub, K_isub, μ_isub, laser_isub, Δt_small):

    #Hmch 
    Hmch_isub = H_isub - torch.sum(torch.sum(μ_isub*laser_isub, -1), -1)
    #laser(t_isub, OMega, fwhm, t_c)
    
    expH = torch.linalg.matrix_exp(-(Hmch_isub*1j + K_isub)*Δt_small)
    
    return expH


def Rtotal(H_t, H_t_Δt, NAC_t, NAC_t_Δt, v_t, v_t_Δt, μ_t, μ_t_Δt, eigvecs_t, eigvecs_t_Δt, laser_t, laser_t_Δt, Δt, t):
    
    eigvecs_t = eigvecs_t.to(torch.complex64)
    eigvecs_t_Δt = eigvecs_t_Δt.to(torch.complex64)
    
    ## NAC
    K_t = torch.sum(torch.sum(NAC_t*(v_t.unsqueeze(1).unsqueeze(1)),3),3)
    K_t_Δt = torch.sum(torch.sum(NAC_t_Δt*(v_t_Δt.unsqueeze(1).unsqueeze(1)),3),3)
    
    Nsubstep = 25
    Δt_small = Δt/Nsubstep
    
    R_total = torch.tile(torch.eye(nstate).unsqueeze(0).type(torch.complex64),[Ntraj,1,1])
    
    for isub in range(1,Nsubstep+1):
        
        H_isub = H_t + (H_t_Δt - H_t)*isub/Nsubstep
        K_isub = K_t + (K_t_Δt - K_t)*isub/Nsubstep
        laser_isub = laser_t + (laser_t_Δt - laser_t)*isub/Nsubstep
        
        ## Dipole moment
        μ_isub = μ_t + (μ_t_Δt - μ_t)*isub/Nsubstep
        t_isub = t + Δt*isub/Nsubstep
        
        expH = propagator(H_isub, K_isub, μ_isub, laser_isub, Δt_small)
        R_total = torch.matmul(expH, R_total)
    
    R_total = torch.matmul( torch.matmul(torch.transpose(torch.conj(eigvecs_t_Δt) ,1,2),R_total), eigvecs_t)
    
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


def update_activatestate(activatestate_index, seqtraj_index, Prob, H, v, m, OMega, random_num):

    laser_hop_width = 1/27.2114
    #OMega = 
#    random_num = np.random.uniform(0,1,Ntraj).reshape(Ntraj,1,1)
    compara_p_r = torch.tensor(Prob.detach().numpy() > random_num).to(torch.int32)
    hop_index = compara_p_r[seqtraj_index, list(1-np.int32(activatestate_index)), activatestate_index ]
    
    ## determine laser or nonadiabatic hopping
    deter_laser = np.int32(np.abs(torch.abs((H[:,1,1] - H[:,0,0])).detach().numpy() - OMega) < laser_hop_width)
    hop_index_laserinduced = hop_index.detach().numpy()*deter_laser
    hop_index_nonadiabatic = hop_index.detach().numpy() - hop_index_laserinduced
    
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
    
    hop_index_True = hop_index_normalhop + hop_index_laserinduced
    
    
    activatestate_index_next = list(np.int32(activatestate_index) + ((-1)**np.int32(activatestate_index))*hop_index_True)
    
    return activatestate_index_next, f_factor


def EDC_decoherence(coefficients, activatestate_index, seqtraj_index, H, eigvals, v, m, Δt):

    c_α = coefficients[seqtraj_index,list(1-np.int32(activatestate_index)),:]
    c_β = coefficients[seqtraj_index,activatestate_index,:]
    
    E_kin = torch.sum(torch.sum(0.5*m*v**2,1),1)
    
    gap = torch.abs(eigvals[:,1] - eigvals[:,0])
#    gap = torch.abs(H[:,1,1] - H[:,0,0])
    
    scale_α = torch.exp(-0.5*Δt*gap*(1/(1 + 0.1/E_kin)))
    c_α_p = scale_α.unsqueeze(1)*c_α
    
    scale_β = torch.sqrt(1 - torch.abs(c_α_p)**2)/torch.abs(c_β)
    c_β_p = scale_β*c_β
    
    coefficients[seqtraj_index,list(1-np.int32(activatestate_index)),:] = c_α_p
    coefficients[seqtraj_index,activatestate_index,:] = c_β_p
    
    return coefficients
