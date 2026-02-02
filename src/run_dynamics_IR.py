#!/usr/bin/env python
from shutil import copyfile, rmtree
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from schnetpack.utils.script_utils.settings import get_environment_provider
from schnetpack.utils import get_loaders
from ase import units
from lct_constants import *
from lct_io import *
from lct_config import *
from electronic import calc_NAC_viaHess, calc_NAC_viaHess_H2
from lct_nuclear_step_RPSH_cayley import * 
from scipy.interpolate import interp2d,interpn



parser = argparse.ArgumentParser(description='select functional',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--nstep_record', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--input', type=str, default='lct.in')
parser.add_argument('--paths', type=str, default='paths.cfg')
args = parser.parse_args()
device = torch.device(f'cuda:{args.gpu_id}')

# ==========================================
#  Load config / paths
# ==========================================

config = load_lct_input(args.input)
paths_cfg = load_lct_input(args.paths)

Ntraj  = int(config.get('ntraj', 300))  
Nbead  = int(config.get('nbead', 1))
Natoms = int(config.get('natoms', 18))
nstate = int(config.get('nstates', 2))
Nring = int(Ntraj / Nbead)

Δt_fs   = float(config.get('dt_fs', 0.5)) 
N_total = int(config.get('nsteps', 6000)) 
Δt      = Δt_fs * scale_t   

Act_state = int(config.get('actstate', 1))

e0    = float(config.get('e0', 5e-3))     
OMega = float(config.get('omega', 0.1183)) 


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ atomic Mass Tensor @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Mass_np = load_mass(Natoms)                           # (Natoms, 3)
Mass = torch.tensor(Mass_np).type(torch.float32)      # 转成 tensor

Mass = Mass.reshape(Natoms * 3, 1) * scale_m
m = Mass.reshape(1, Natoms, 3)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ schnet inputs format // dict #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


path_initconds = paths_cfg.get('initconds_path', './Initconds')
topology_file  = paths_cfg.get('topology_file', f"{path_initconds}/initdb_dict_{Ntraj}.pth")
bbdict = torch.load(topology_file, map_location=device)
bb = {k: v.to(device) for k, v in bbdict.items()}



atomic_numbers_ = bb['_atomic_numbers']
cell_ = bb['_cell']
cell_offset_ = bb['_cell_offset']
neighbors_ = bb['_neighbors']
neighbor_mask_ = bb['_neighbor_mask']
atom_mask_ = bb['_atom_mask']

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@          model path and model loading        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ PESNN may have two models (hess and nonhess) @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



pes_root = paths_cfg.get('pes_model_root', './Model/PES')
pes_model_file = f"{pes_root.rstrip('/')}/best_model"

model = torch.load(pes_model_file, map_location="cpu").to(device)
model = nn.DataParallel(model)

                                       ## for NN parall
for p in model.module.output_modules[0].output_dict.keys():                      ## for NN parall
    model.module.output_modules[0].output_dict[p].return_hessian = [True,2,1,1,1,False]  ## for NN parall

### Dipole meomet model path
dipole_dir = paths_cfg.get('dipole_model_dir', './Model/Dipole')
dipole_model_file = f"{dipole_dir.rstrip('/')}/best_model"

model_dp = torch.load(dipole_model_file, map_location="cpu").to(device)
model_dp = nn.DataParallel(model_dp)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    initconds file      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ for all trajs (Ntrajs) @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
paths_initcoord = path_initconds
paths_initveloc = path_initconds

x_init_, v_init_ = get_initconds(paths_initcoord, paths_initveloc)
x_init, v_init = torch.tensor(x_init_).type(torch.float32),  torch.tensor(v_init_).type(torch.float32)


#random_num_all = np.expand_dims(np.loadtxt('./random_num.txt'),-1)
random_num_all = load_random_numbers()


# 1. 设置固定的随机种子（保证可复现性）
torch.manual_seed(42)  # 设置固定随机种子

# 2. 动态生成均匀分布的随机数，范围是 [0, 1)
random_num_all = torch.rand(Nring, Nbead, 1).to(device)

# 3. 后续计算中使用随机数（例如，跳跃决策）
random_num = random_num_all[nstep, :, 0] 




##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Laser induced FSSH  (SHARC methods)   @@@@@@@@@@@@@@@@@@@
dict_list = ['x_t', 'v_t', 'H_t', 'F_t', 'NAC_t', 'Hess_t',
 'μ_t', 'dμdx_t', 'E_laser_t',
 'Hmch_t', 'eigvals_t', 'eigvecs_t', 'Fdiag_t',
 'x_t_ca', 'v_t_ca', 'H_t_ca', 'F_t_ca', 'NAC_t_ca', 'Hess_t_ca',
 'μ_t_ca', 'dμdx_t_ca', 'E_laser_t_ca', 
 'Hmch_t_ca', 'eigvals_t_ca','eigvecs_t_ca',
 'coefficient_t']


savedir = paths_cfg.get('savedir', './TRAJ_lct_IR')


seqring_index = [II for II in range(Nring)]
seqtraj_index = [II for II in range(Ntraj)]
coefficient_init =torch.tile(torch.transpose(torch.nn.functional.one_hot(torch.tensor([Act_state]),2),0,1).unsqueeze(0), [Nring,1,1]).type(torch.complex64)



###@@@@@@@@@@@@@@@@@@@@@ vars for free RP step @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#Mat_ring = torch.tensor( np.tile(np.expand_dims(Get_RingMatrix(Nbead),0), [Nring,1,1]) ).type(torch.float32)
#C_mat = Cmatrix(Nbead)
#C_mat_eigval = Cmatrix_eigvals(Nbead)

## frequency of beads connection
#Temper = 300 #298.15 ## temperature for RPMD ( units: K)
Kb = 3.1668089604214317e-06  ## a. u.
#ωn = Nbead*Kb*Temper

## cayley transform matrix
#cayA_T = CayleyAT_transform(Nring, Nbead, C_mat_eigval,ωn,Δt)

## split parts number
Nsplit_traj = 1
Nsplit_ca = 1

nstep=args.nstep_record
t=nstep*Δt

if nstep==0:
    ### intializing electronic state and x, v
    ### coefficent         (Nring) 
    ### x, v             (Ntraj)
    ## active state for rings (Nring)
    activatestate_index_t_ca = [Act_state]*Nring
    ## extend activatestate_index_t_ca to activatestate_index_t (Ntraj)
    activatestate_index_t = [ I for I in activatestate_index_t_ca for J in range(Nbead) ]

    coefficient_t = coefficient_init
    x_t, v_t = x_init, v_init
    
    
    ### calc Hmch and force for all trajs (Ntraj)
    H_t, F_t, NAC_t, Hess_t = Hamiltonian_output_nac(x_t, Nsplit_traj, model,
                                    atomic_numbers_,
                                    cell_offset_,
                                    cell_,
                                    neighbors_,
                                    neighbor_mask_,
                                    atom_mask_,m,device)
    
    
    ### calc dipole and its gradients for all trajs (Ntraj)
    μ_t, dμdx_t = Dipole_Moment(x_t, Nsplit_traj, model_dp, 
                        atomic_numbers_,
                        cell_offset_,
                        cell_,
                        neighbors_,
                        neighbor_mask_,
                        atom_mask_,m,device)
    

    ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ LCT fields @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ### calc VR-excited LCT fields using ring centrals (CA)
    ### x_t_ca = { <x>_1, <x>_2, <x>_3, ... , <x>_Nring }; <x>_k is beads average value of k-th ring
    ### v_t_ca = { <v>_1, <v>_2, <v>_3, ... , <v>_Nring }
    ### E_laser_t_ca is averaged on all ring centrals
    x_t_ca = torch.mean(x_t.reshape(Nring, Nbead, Natoms, 3), 1)
    v_t_ca = torch.mean(v_t.reshape(Nring, Nbead, Natoms, 3), 1)
    
    bondCN1 = ( torch.sqrt(torch.sum((x_t_ca[:,16,:] - x_t_ca[:,0,:])**2, -1)).unsqueeze(1).unsqueeze(1) ).detach()
    bondCN2 = ( torch.sqrt(torch.sum((x_t_ca[:,17,:] - x_t_ca[:,1,:])**2, -1)).unsqueeze(1).unsqueeze(1) ).detach()
    sfunc = fmulti_funct_circle(bondCN1, bondCN2)
    dihCNNC = calcdihCNC(x_t_ca.cpu().detach().numpy())
    Q = np.concatenate((np.minimum(bondCN1.squeeze(-1).numpy(), 3.45*scale_L),
                    np.minimum(bondCN2.squeeze(-1).numpy(), 3.45*scale_L),
                    np.expand_dims(dihCNNC,-1) ),axis=1)
    
    dXdQ = Jacob_DXDQ(xyzcoord_Bohr, Q)
    
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  Dipole of CA >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    μ_t_ca, dμdx_t_ca = Dipole_Moment(x_t_ca, Nsplit_ca, model_dp, 
                                  atomic_numbers_,
                                  cell_offset_,
                                  cell_,
                                  neighbors_,
                                  neighbor_mask_,
                                  atom_mask_,m,device)
    
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    vca_masscenter = ( torch.sum(v_t_ca*m, 1)/torch.sum(m[:,:,0]) ).unsqueeze(1)
    Pq = torch.sum(torch.sum( dXdQ.reshape(Nring, Natoms, 3 ,2) * (m*(v_t_ca-vca_masscenter)).unsqueeze(-1), -2), -2).unsqueeze(1)
    dμdq = torch.sum(torch.sum(dμdx_t_ca[:,1,1].unsqueeze(-1) * dXdQ.reshape(Nring, Natoms, 3 ,2).unsqueeze(1), -2), -2)
    
    #fields_t = e0*sfunc*torch.sum(Pq*dμdq,-1).unsqueeze(1)
    #@@@@@@@@@@@@@ new methond LCT : e0*<s*dμdQ><s*Pq> @@ compare to QD @@@@@@@@@@@@@@@@@@@@@@@@@
    fields_t = e0*torch.tile(torch.sum( torch.mean(sfunc*Pq,0) * torch.mean(sfunc*dμdq,0), -1).unsqueeze(0).unsqueeze(0), [Nring,1,1])

    #E_laser_t = torch.tile(torch.mean(fields_t.detach(), 0).unsqueeze(0), [Ntraj,1,1])
    E_laser_t = torch.tile(torch.sum(fields_t.detach(), 0).unsqueeze(0)/Nring, [Ntraj,1,1])
    E_laser_t_ca = torch.tile(torch.sum(fields_t.detach(), 0).unsqueeze(0)/Nring, [Nring,1,1])
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Diag Force of all trajs @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ## MCH Ham matrix for all trajs (Ntraj) 
    Hmch_t = H_MCH(H_t,μ_t.detach(), torch.tile(E_laser_t.unsqueeze(1).unsqueeze(1), [1,nstate,nstate,1,1]))
    ## Get U_matrix for all trajs   (Ntraj)
    eigvals_t, eigvecs_t = H_Diagonal(Hmch_t)
    ## Get diag forces for all trajs (Ntraj)
    Fdiag_t = F_Diagonal(F_t, dμdx_t, 
                         torch.tile(E_laser_t.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), [1,nstate,nstate,1,Natoms,3]),
                         eigvecs_t)
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  calc the H and NAC for ring CA (to propagator Rtot) (Nring) @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    H_t_ca, F_t_ca, NAC_t_ca, Hess_t_ca = Hamiltonian_output_nac(x_t_ca, Nsplit_ca, model,
                                             atomic_numbers_,
                                             cell_offset_,
                                             cell_,
                                             neighbors_,
                                             neighbor_mask_,
                                             atom_mask_,m,device)
    
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>> MCH Ham matrix for all rings (Nring)
    Hmch_t_ca = H_MCH(H_t_ca,μ_t_ca.detach(), torch.tile(E_laser_t_ca.unsqueeze(1).unsqueeze(1), [1,nstate,nstate,1,1]))
    ##>>>>>>>>>>>>>>>>>>>>>>>>>>> Get U_matrix for all rings  (Nring)
    eigvals_t_ca, eigvecs_t_ca = H_Diagonal(Hmch_t_ca)
    
    
    ## save data for both trajs_var and ringca_var
    datatensor_tstep = x_t, v_t, H_t, F_t, NAC_t, Hess_t,\
    μ_t, dμdx_t, E_laser_t,\
    Hmch_t, eigvals_t, eigvecs_t, Fdiag_t,\
    x_t_ca, v_t_ca, H_t_ca, F_t_ca, NAC_t_ca, Hess_t_ca,\
    μ_t_ca, dμdx_t_ca, E_laser_t_ca,\
    Hmch_t_ca, eigvals_t_ca,eigvecs_t_ca,\
    coefficient_t

    datadict_tstep = {}
    i = 0
    for keys in dict_list:
        datadict_tstep[keys] = datatensor_tstep[i]
        i += 1
    save_step_dict(savedir, nstep, datadict_tstep)
    
    ## save istate
    save_occstate(savedir, nstep, activatestate_index_t_ca)

    
    ## save react hopping traj's index
    #np.savetxt(f"{savedir}/react_occstate_{nstep}.txt",
    #          np.int64(np.array([])),
    #          delimiter='  ',
    #          fmt="%d")
    
else:
    ##@@@@@@@@@@@@@@@@@@@    Random Number for hopings     @@@@@@@@@@@@@@@@@@@@@@@
    random_num = random_num_all[:Nring,nstep:nstep+1,0:1]
    ##@@@@@@@@@@@@@@@@@@ Load datadict from last step @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    activatestate_index_t_ca = load_occstate(savedir, nstep - 1)

    activatestate_index_t = [ I for I in activatestate_index_t_ca for J in range(Nbead) ]
    datadict_tstep = load_step_dict(savedir, nstep - 1, device=device)
    
    x_t, v_t, H_t, F_t, NAC_t, Hess_t, μ_t, dμdx_t, E_laser_t = datadict_tstep['x_t'], datadict_tstep['v_t'],\
    datadict_tstep['H_t'], datadict_tstep['F_t'], datadict_tstep['NAC_t'], datadict_tstep['Hess_t'],\
    datadict_tstep['μ_t'], datadict_tstep['dμdx_t'], datadict_tstep['E_laser_t']
    
    Hmch_t, eigvals_t, eigvecs_t, Fdiag_t = datadict_tstep['Hmch_t'], datadict_tstep['eigvals_t'],\
    datadict_tstep['eigvecs_t'], datadict_tstep['Fdiag_t']
    
    x_t_ca,v_t_ca,H_t_ca,F_t_ca,NAC_t_ca,Hess_t_ca,μ_t_ca,dμdx_t_ca,E_laser_t_ca = datadict_tstep['x_t_ca'], datadict_tstep['v_t_ca'],\
    datadict_tstep['H_t_ca'], datadict_tstep['F_t_ca'], datadict_tstep['NAC_t_ca'], datadict_tstep['Hess_t_ca'],\
    datadict_tstep['μ_t_ca'], datadict_tstep['dμdx_t_ca'], datadict_tstep['E_laser_t_ca']
    
    Hmch_t_ca, eigvals_t_ca, eigvecs_t_ca = datadict_tstep['Hmch_t_ca'], datadict_tstep['eigvals_t_ca'],datadict_tstep['eigvecs_t_ca']
    coefficient_t = datadict_tstep['coefficient_t']
    
    ## hopping traj's idx (accumulated)
    #reacttraj_t = np.loadtxt(f"{savedir}/react_occstate_{nstep-1}.txt", dtype=int)
    #nonreacttraj_t = np.setdiff1d( np.int64(seqtraj_index), reacttraj_t) 
    
    ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@     calc all trajs (Ntrajs)    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ STOP At Here !!!!! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
    ## rF_t :  ring force
    #rF_t = Ring_forces(x_t, Mat_ring, m, Temper)
    #x_t_Δt = vverlet_X(x_t, v_t,
    #                   Fdiag_t[seqtraj_index,activatestate_index_t,:], rF_t,
    #                   m, Δt)
    
    #x_t_Δt = vverlet_X(x_t, v_t,
    #                   Fdiag_t[seqtraj_index,activatestate_index_t,:],
    #                   m, Δt)
    
    x_t_Δt = vverlet_X(x_t, v_t,
                       Fdiag_t[seqtraj_index,activatestate_index_t,:],
                       m, Δt)
    
    ## half step Veolcity
    #v_t_hΔt = veloc_halfstep(v_t, Fdiag_t[seqtraj_index,activatestate_index_t,:], m, Δt)
    
    ## full step Free RP
    #x_t_Δt, v_t_Δt = free_RPstep(Nring, Nbead, x_t, v_t_hΔt, C_mat, cayA_T)
    
    
    ### calc Hmch and force for all trajs (Ntraj)
    H_t_Δt, F_t_Δt, NAC_t_Δt, Hess_t_Δt = Hamiltonian_output_nac(x_t_Δt,Nsplit_traj, model,
                                        atomic_numbers_,
                                        cell_offset_,
                                        cell_,
                                        neighbors_,
                                        neighbor_mask_,
                                        atom_mask_,m,device)
    
    ### calc dipole and its gradients for all trajs (Ntraj)
    μ_t_Δt, dμdx_t_Δt = Dipole_Moment(x_t_Δt, Nsplit_traj, model_dp, 
                                      atomic_numbers_,
                                      cell_offset_,
                                      cell_,
                                      neighbors_,
                                      neighbor_mask_,
                                      atom_mask_,m,device)
    
    
    ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ LCT fields @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ### calc VR-excited LCT fields using ring centrals (CA)
    ### x_t_ca = { <x>_1, <x>_2, <x>_3, ... , <x>_Nring }; <x>_k is beads average value of k-th ring
    ### v_t_ca = { <v>_1, <v>_2, <v>_3, ... , <v>_Nring }
    ### E_laser_t_ca is averaged on all ring centrals
    
    x_t_Δt_ca = torch.mean(x_t_Δt.reshape(Nring, Nbead, Natoms, 3), 1)
    v_t_Δt_ca = torch.mean(v_t.reshape(Nring, Nbead, Natoms, 3), 1)
    
    bondCN1 = ( torch.sqrt(torch.sum((x_t_Δt_ca[:,16,:] - x_t_Δt_ca[:,0,:])**2, -1)).unsqueeze(1).unsqueeze(1) ).detach()
    bondCN2 = ( torch.sqrt(torch.sum((x_t_Δt_ca[:,17,:] - x_t_Δt_ca[:,1,:])**2, -1)).unsqueeze(1).unsqueeze(1) ).detach()
    sfunc = fmulti_funct_circle(bondCN1, bondCN2)
    dihCNNC = calcdihCNC(x_t_Δt_ca.cpu().detach().numpy())
    Q = np.concatenate((np.minimum(bondCN1.squeeze(-1).numpy(), 3.45*scale_L),
                    np.minimum(bondCN2.squeeze(-1).numpy(), 3.45*scale_L),
                    np.expand_dims(dihCNNC,-1) ),axis=1)
    
    dXdQ = Jacob_DXDQ(xyzcoord_Bohr, Q)
    
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  Dipole of CA >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    μ_t_Δt_ca, dμdx_t_Δt_ca = Dipole_Moment(x_t_Δt_ca, Nsplit_ca, model_dp, 
                                  atomic_numbers_,
                                  cell_offset_,
                                  cell_,
                                  neighbors_,
                                  neighbor_mask_,
                                  atom_mask_,m,device)
    
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    vca_masscenter = ( torch.sum(v_t_Δt_ca*m, 1)/torch.sum(m[:,:,0]) ).unsqueeze(1)
    Pq = torch.sum(torch.sum( dXdQ.reshape(Nring, Natoms, 3 ,2) * (m*(v_t_Δt_ca-vca_masscenter)).unsqueeze(-1), -2), -2).unsqueeze(1)
    dμdq = torch.sum(torch.sum(dμdx_t_Δt_ca[:,1,1].unsqueeze(-1) * dXdQ.reshape(Nring, Natoms, 3 ,2).unsqueeze(1), -2), -2)
    
    #fields_t_Δt = e0*sfunc*torch.sum(Pq*dμdq,-1).unsqueeze(1)
    ##@@@@@@@@@@@ new method LCT 
    fields_t_Δt = e0*torch.tile(torch.sum( torch.mean(sfunc*Pq,0) * torch.mean(sfunc*dμdq,0), -1).unsqueeze(0).unsqueeze(0), [Nring,1,1])

    #E_laser_t = torch.tile(torch.mean(fields_t.detach(), 0).unsqueeze(0), [Ntraj,1,1])
    E_laser_t_Δt = torch.tile(torch.sum(fields_t_Δt.detach(), 0).unsqueeze(0)/Nring, [Ntraj,1,1])
    E_laser_t_Δt_ca = torch.tile(torch.sum(fields_t_Δt.detach(), 0).unsqueeze(0)/Nring, [Nring,1,1])
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ calc the diag Force and propagate the veloc @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@        calc for all trajs (Ntrajs) @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    Hmch_t_Δt = H_MCH(H_t_Δt,μ_t_Δt.detach(), torch.tile(E_laser_t_Δt.unsqueeze(1).unsqueeze(1), [1,nstate,nstate,1,1]) )
    eigvals_t_Δt, eigvecs_t_Δt = H_Diagonal(Hmch_t_Δt)
    ## phase correction
    eigvecs_t_Δt, eigvals_t_Δt, P_project_oth  = Interpolation_U(Hmch_t, eigvecs_t, Hmch_t_Δt)
    ## calc diagonal forces
    Fdiag_t_Δt = F_Diagonal(F_t_Δt, dμdx_t_Δt, 
                            torch.tile(E_laser_t_Δt.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), [1,nstate,nstate,1,Natoms,3]), 
                            eigvecs_t_Δt)
    
    
    #rF_t_Δt = Ring_forces(x_t_Δt, Mat_ring, m, Temper)
    #v_t_Δt = vverlet_V(v_t, 
    #                   Fdiag_t[seqtraj_index,activatestate_index_t,:],
    #                   Fdiag_t_Δt[seqtraj_index,activatestate_index_t,:],
    #                   rF_t, rF_t_Δt,
    #                   m, Δt)
    
    ## half step Veolcity
    #v_t_Δt = veloc_halfstep(v_t_Δt, Fdiag_t_Δt[seqtraj_index,activatestate_index_t,:], m, Δt)
    v_t_Δt = vverlet_V(v_t, 
                       Fdiag_t[seqtraj_index,activatestate_index_t,:],
                       Fdiag_t_Δt[seqtraj_index,activatestate_index_t,:],
                       m, Δt)
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
    ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  calc the H and NAC for ring CA (to propagator Rtot) (Nring) @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    v_t_Δt_ca = torch.mean(v_t_Δt.reshape(Nring, Nbead, Natoms, 3), 1)
    
    H_t_Δt_ca, F_t_Δt_ca, NAC_t_Δt_ca, Hess_t_Δt_ca = Hamiltonian_output_nac(x_t_Δt_ca, Nsplit_ca, model,
                                         atomic_numbers_,
                                         cell_offset_,
                                         cell_,
                                         neighbors_,
                                         neighbor_mask_,
                                         atom_mask_,m,device)
    
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>> MCH Ham matrix for all rings (Nring)
    Hmch_t_Δt_ca = H_MCH(H_t_Δt_ca,μ_t_Δt_ca.detach(), torch.tile(E_laser_t_Δt_ca.unsqueeze(1).unsqueeze(1), [1,nstate,nstate,1,1]))
    ##>>>>>>>>>>>>>>>>>>>>>>>>>>> Get U_matrix for all rings  (Nring)
    eigvals_t_Δt_ca, eigvecs_t_Δt_ca = H_Diagonal(Hmch_t_Δt_ca)
    ## phase correction
    eigvecs_t_Δt_ca, eigvals_t_Δt_ca, P_project_oth_ca  = Interpolation_U(Hmch_t_ca, eigvecs_t_ca, Hmch_t_Δt_ca)
    
    
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ calc the Rtotal for propagating coefficent  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ calc for all central (Nrings) @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    R_total = Rtotal(H_t_ca, H_t_Δt_ca, NAC_t_ca, NAC_t_Δt_ca, v_t_ca, v_t_Δt_ca, μ_t_ca, μ_t_Δt_ca, eigvecs_t_ca, eigvecs_t_Δt_ca, 
                 torch.tile(E_laser_t_ca.unsqueeze(1).unsqueeze(1), [1,nstate,nstate,1,1]), 
                 torch.tile(E_laser_t_Δt_ca.unsqueeze(1).unsqueeze(1), [1,nstate,nstate,1,1]),
                 Δt, t)
    
    #R_total = Rtotal(H_t, H_t_Δt, NAC_t, NAC_t_Δt, v_t, v_t_Δt, μ_t, μ_t_Δt, eigvecs_t, eigvecs_t_Δt,Δt, t, laser, OMega, fwhm, t_c)
    coefficient_t_Δt = update_coefficient(R_total,coefficient_t)
    Prob = Prob_hop(coefficient_t, coefficient_t_Δt, R_total)
    
    activatestate_index_t_Δt_ca, f_factor_ca = update_activatestate(activatestate_index_t_ca, seqring_index, Prob, H_t_Δt_ca, v_t_Δt_ca, m, OMega,random_num)
    activatestate_index_t_Δt = [ I for I in activatestate_index_t_Δt_ca for J in range(Nbead) ]
    f_factor = torch.tile(f_factor_ca,[1,Nbead,1]).reshape(Nring*Nbead,1,1)
    
    v_t_Δt_ca = v_t_Δt_ca*f_factor_ca
    v_t_Δt = v_t_Δt*f_factor

    ## Energy-based decoherence
    coefficient_t_Δt = EDC_decoherence(coefficient_t_Δt, activatestate_index_t_Δt_ca, seqring_index, H_t_Δt_ca, eigvals_t_Δt_ca, v_t_Δt_ca, m, Δt)
    
    ##
    ## @@@@@@@@@@@@@@@@@@@@@@@ save data for nowstep (t+Δt) @@########################################
    datatensor_tstep = x_t_Δt, v_t_Δt, H_t_Δt, F_t_Δt, NAC_t_Δt, Hess_t_Δt,\
    μ_t_Δt, dμdx_t_Δt, E_laser_t_Δt,\
    Hmch_t_Δt, eigvals_t_Δt, eigvecs_t_Δt, Fdiag_t_Δt,\
    x_t_Δt_ca, v_t_Δt_ca, H_t_Δt_ca, F_t_Δt_ca, NAC_t_Δt_ca, Hess_t_Δt_ca,\
    μ_t_Δt_ca, dμdx_t_Δt_ca, E_laser_t_Δt_ca,\
    Hmch_t_Δt_ca, eigvals_t_Δt_ca,eigvecs_t_Δt_ca,\
    coefficient_t_Δt
    
    datadict_tstep = {}
    i = 0
    for keys in dict_list:
        datadict_tstep[keys] = datatensor_tstep[i]
        i += 1
    save_step_dict(savedir, nstep, datadict_tstep)

    
    ## save istate for nowstep (t+Δt)
    save_occstate(savedir, nstep, activatestate_index_t_Δt_ca)


    


