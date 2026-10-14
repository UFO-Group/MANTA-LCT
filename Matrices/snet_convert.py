import numpy as np
import torch

def schnet_input(positions_x, _atomic_numbers_, _cell_offset_, _cell_, _neighbors_, _neighbor_mask_, _atom_mask_):
    input_snet = {}
    input_snet['_positions'] = positions_x
    input_snet['_atomic_numbers'] = _atomic_numbers_
    input_snet['_cell_offset'] = _cell_offset_
    input_snet['_cell'] = _cell_
    input_snet['_neighbors'] = _neighbors_
    input_snet['_neighbor_mask'] = _neighbor_mask_
    input_snet['_atom_mask'] = _atom_mask_
    
    
    return input_snet