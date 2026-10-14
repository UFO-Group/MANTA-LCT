import torch
import numpy as np



def calc_NAC_viaHess(energys_,forces_,hess_): ## for two states :(i,j) ---> NAC_ij
    
    ## energys_ : torch ( Ntraj x Nstate)
    ## forces_ : torch (Ntraj x Nstate x Natoms x 3)
    ## hess_ : torch  (Ntraj x Nstate x 3Natom x 3Natom)
    
    Ntraj = energys_.shape[0]
    Natoms = forces_.shape[2]
    
    istate = 0
    jstate = 1
    
    Hi = hess_[:,istate]
    Hj = hess_[:,jstate]
    
    dE = energys_[:,istate]-energys_[:,jstate]
    
    forces_ = -forces_
    GiGi = forces_[:,istate].reshape(Ntraj,Natoms*3,1)*forces_[:,istate].reshape(Ntraj,1,Natoms*3)
    GiGj = forces_[:,istate].reshape(Ntraj,Natoms*3,1)*forces_[:,jstate].reshape(Ntraj,1,Natoms*3)
    GjGi = forces_[:,jstate].reshape(Ntraj,Natoms*3,1)*forces_[:,istate].reshape(Ntraj,1,Natoms*3)
    GjGj = forces_[:,jstate].reshape(Ntraj,Natoms*3,1)*forces_[:,jstate].reshape(Ntraj,1,Natoms*3)
    
    G_diff = 0.5*(forces_[:,istate] -forces_[:, jstate])
    G_diff2= G_diff.reshape(Ntraj,Natoms*3,1)*G_diff.reshape(Ntraj,1,Natoms*3)
    
    #dH_2_ij = 0.5*(dE.unsqueeze(1).unsqueeze(2)*(Hi-Hj) + GiGi + GjGj - 2*GiGj)
    #magnitude = dH_2_ij/2-G_diff2
    
    ## @@@@@@@@@@@@@@@@@@@@@ fixed bug for magnitude
    #magnitude = 0.5*0.5*(dE.unsqueeze(1).unsqueeze(2)*(Hi-Hj))
    magnitude = 0.5*0.5*(dE.unsqueeze(1).unsqueeze(2)*(Hi-Hj)) + (GiGi + GjGj - 2*GiGj)/4 - G_diff2
    
    u,s,vh = np.linalg.svd(magnitude.numpy())
    ev, ew = torch.tensor(vh[:,0]), torch.tensor(s[:,0])
    
    #u,s,vh = torch.linalg.svd(magnitude)
    #ev, ew = vh[:,0], s[:,0]
    
    hopping_direction = ev.reshape(Ntraj,Natoms,3)  ## for NAC[i,j]
    hopping_magnitude = ew.reshape(Ntraj,1,1)
    
    magnitude = torch.sqrt(hopping_magnitude)/torch.abs(dE.unsqueeze(1).unsqueeze(2)) #/2
    
    threshold_dE = 0.04 #0.04
    
    Mask_nac = (torch.abs(dE.unsqueeze(1).unsqueeze(2)) < threshold_dE).type(torch.float32)
    
    nacij = hopping_direction*magnitude*Mask_nac  ## for NAC_ij
    
    nstate = 2
    NACdr = torch.zeros([Ntraj,nstate,nstate,Natoms,3])
    NACdr[:,0:1,1:2] = nacij.unsqueeze(1).unsqueeze(1)
    NACdr[:,1:2,0:1] = -nacij.unsqueeze(1).unsqueeze(1)
    NACdr[:,0:1,0:1] = torch.zeros_like(nacij.unsqueeze(1).unsqueeze(1))
    NACdr[:,1:2,1:2] = torch.zeros_like(nacij.unsqueeze(1).unsqueeze(1))
    
    return NACdr



def calc_NAC_viaHess_H2(energys_,forces_,hess_): ## for two states :(i,j) ---> NAC_ij
    
    ## energys_ : torch ( Ntraj x Nstate)
    ## forces_ : torch (Ntraj x Nstate x Natoms x 3)
    ## hess_ : torch  (Ntraj x Nstate x 3Natom x 3Natom)
    
    Ntraj = energys_.shape[0]
    Natoms = forces_.shape[2]
    
    istate = 0
    jstate = 1
    
    Hi = hess_[:,istate]
    Hj = hess_[:,jstate]
    
    dE = energys_[:,istate]-energys_[:,jstate]
    
    #forces_ = -forces_
    #GiGi = forces_[:,istate].reshape(Ntraj,Natoms*3,1)*forces_[:,istate].reshape(Ntraj,1,Natoms*3)
    #GiGj = forces_[:,istate].reshape(Ntraj,Natoms*3,1)*forces_[:,jstate].reshape(Ntraj,1,Natoms*3)
    #GjGi = forces_[:,jstate].reshape(Ntraj,Natoms*3,1)*forces_[:,istate].reshape(Ntraj,1,Natoms*3)
    #GjGj = forces_[:,jstate].reshape(Ntraj,Natoms*3,1)*forces_[:,jstate].reshape(Ntraj,1,Natoms*3)
    
    #G_diff = 0.5*(forces_[:,istate] -forces_[:, jstate])
    #G_diff2= G_diff.reshape(Ntraj,Natoms*3,1)*G_diff.reshape(Ntraj,1,Natoms*3)
    
    dH_2_ij = Hi-Hj
    magnitude = dH_2_ij
    
    #u,s,vh = np.linalg.svd(magnitude.numpy())
    #ev, ew = torch.tensor(vh[:,0]), torch.tensor(s[:,0])
    u,s,vh = torch.linalg.svd(magnitude)
    ev, ew = vh[:,0], s[:,0]
    
    hopping_direction = ev.reshape(Ntraj,Natoms,3)  ## for NAC[i,j]
    hopping_magnitude = ew.reshape(Ntraj,1,1)
    
    magnitude = hopping_magnitude/torch.abs(dE.unsqueeze(1).unsqueeze(2))/2
    
    threshold_dE = 0.04 #0.04
    
    Mask_nac = (torch.abs(dE.unsqueeze(1).unsqueeze(2)) < threshold_dE).type(torch.float32)
    
    nacij = hopping_direction*magnitude*Mask_nac  ## for NAC_ij
    
    nstate = 2
    NACdr = torch.zeros([Ntraj,nstate,nstate,Natoms,3])
    NACdr[:,0:1,1:2] = nacij.unsqueeze(1).unsqueeze(1)
    NACdr[:,1:2,0:1] = -nacij.unsqueeze(1).unsqueeze(1)
    NACdr[:,0:1,0:1] = torch.zeros_like(nacij.unsqueeze(1).unsqueeze(1))
    NACdr[:,1:2,1:2] = torch.zeros_like(nacij.unsqueeze(1).unsqueeze(1))
    
    return NACdr


def calc_NAC_viaHess(energys_,forces_,hess_): ## for two states :(i,j) ---> NAC_ij
    
    ## energys_ : torch ( Ntraj x Nstate)
    ## forces_ : torch (Ntraj x Nstate x Natoms x 3)
    ## hess_ : torch  (Ntraj x Nstate x 3Natom x 3Natom)
    
    Nsamp = energys_.shape[0]
    Natoms = forces_.shape[2]
    
    istate = 0
    jstate = 1
    
    Hi = hess_[:,istate]
    Hj = hess_[:,jstate]
    
    dE = energys_[:,istate]-energys_[:,jstate]
    
    forces_ = -forces_
    GiGi = forces_[:,istate].reshape(Nsamp,Natoms*3,1)*forces_[:,istate].reshape(Nsamp,1,Natoms*3)
    GiGj = forces_[:,istate].reshape(Nsamp,Natoms*3,1)*forces_[:,jstate].reshape(Nsamp,1,Natoms*3)
    GjGi = forces_[:,jstate].reshape(Nsamp,Natoms*3,1)*forces_[:,istate].reshape(Nsamp,1,Natoms*3)
    GjGj = forces_[:,jstate].reshape(Nsamp,Natoms*3,1)*forces_[:,jstate].reshape(Nsamp,1,Natoms*3)
    
    G_diff = 0.5*(forces_[:,istate] -forces_[:, jstate])
    G_diff2= G_diff.reshape(Nsamp,Natoms*3,1)*G_diff.reshape(Nsamp,1,Natoms*3)
    
    #dH_2_ij = 0.5*(dE.unsqueeze(1).unsqueeze(2)*(Hi-Hj) + GiGi + GjGj - 2*GiGj)
    #magnitude = dH_2_ij/2-G_diff2
    
    magnitude = 0.5*0.5*(dE.unsqueeze(1).unsqueeze(2)*(Hi-Hj))
    
    u,s,vh = np.linalg.svd(magnitude.numpy())
    ev, ew = torch.tensor(vh[:,0]), torch.tensor(s[:,0])
    
    #u,s,vh = torch.linalg.svd(magnitude)
    #ev, ew = vh[:,0], s[:,0]
    
    hopping_direction = ev.reshape(Nsamp,Natoms,3)  ## for NAC[i,j]
    hopping_magnitude = ew.reshape(Nsamp,1,1)
    
    magnitude = torch.sqrt(hopping_magnitude)/torch.abs(dE.unsqueeze(1).unsqueeze(2)) #/2
    
    threshold_dE = 0.04 #0.04
    
    Mask_nac = (torch.abs(dE.unsqueeze(1).unsqueeze(2)) < threshold_dE).type(torch.float32)
    
    nacij = hopping_direction*magnitude*Mask_nac  ## for NAC_ij
    
    nstate = 2
    NACdr = torch.zeros([Nsamp,nstate,nstate,Natoms,3])
    NACdr[:,0:1,1:2] = nacij.unsqueeze(1).unsqueeze(1)
    NACdr[:,1:2,0:1] = -nacij.unsqueeze(1).unsqueeze(1)
    NACdr[:,0:1,0:1] = torch.zeros_like(nacij.unsqueeze(1).unsqueeze(1))
    NACdr[:,1:2,1:2] = torch.zeros_like(nacij.unsqueeze(1).unsqueeze(1))
    
    return NACdr



  