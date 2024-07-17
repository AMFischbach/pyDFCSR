import numpy as np

from utility_functions import full_path
from beam import Beam
from lattice import Lattice
from step_snapshot import Step_Snapshot

"""
Module Name: parameter_initialization.py

Initializes some dictionaries for CSR3D and creates a Beam objects
"""

def init_attributes(input_dict = {}):
    """
    Given the input dictionary, creates and returns the integration parameter and 
    csr parameter dictionaries along with the Beam, Lattice, and DF_Tracker objects. 
    """
    
    integration_params = init_integration_params(**input_dict.get("CSR_integration", {}))
    CSR_mesh_params = init_CSR_params(**input_dict.get("CSR_computation", {}))
    

    beam = Beam(input_dict["input_beam"])
    lattice = Lattice(input_dict["input_lattice"], beam.init_energy)

    step_snaphots = init_snapshots(lattice.total_steps, input_dict)
    statistics = init_stats(lattice.total_steps)

    return statistics, integration_params, CSR_mesh_params, lattice, beam, step_snaphots

def init_stats(total_steps):
    """
    Initialize the statitics dictionary, which stores the twiss of the beam at each step
    (also a dictionary) and other various statistics. All statistics are a np array which have an
    element for each 'step' in the lattice.
    """

    # The number of steps in the lattice
    Nstep = total_steps

    # Preallocate size based upon Nstep of np arrays for speed
    statistics = {}
    statistics['twiss'] = {'alpha_x': np.zeros(Nstep),
                                'beta_x': np.zeros(Nstep),
                                'gamma_x': np.zeros(Nstep),
                                'emit_x': np.zeros(Nstep),
                                'eta_x': np.zeros(Nstep),
                                'etap_x': np.zeros(Nstep),
                                'norm_emit_x': np.zeros(Nstep),
                                'alpha_y': np.zeros(Nstep),
                                'beta_y': np.zeros(Nstep),
                                'gamma_y': np.zeros(Nstep),
                                'emit_y': np.zeros(Nstep),
                                'eta_y': np.zeros(Nstep),
                                'etap_y': np.zeros(Nstep),
                                'norm_emit_y': np.zeros(Nstep)}

    statistics['slope'] = np.zeros((Nstep, 2))
    statistics['sigma_x'] = np.zeros(Nstep)
    statistics['sigma_z'] = np.zeros(Nstep)
    statistics['sigma_energy'] = np.zeros(Nstep)
    statistics['mean_x']  = np.zeros(Nstep)
    statistics['mean_z'] = np.zeros(Nstep)
    statistics['mean_energy'] = np.zeros(Nstep)

    return statistics

def init_integration_params(n_formation_length = 4, zbins = 200, xbins = 200):
    """ Initializes the integration_params dictionary"""

    keys = ["n_formation_length", "zbins", "xbins"]
    values = [n_formation_length, zbins, xbins]

    integration_params = {k: v for k, v in zip(keys, values)}
    
    return integration_params

def init_CSR_params(workdir = '.', apply_CSR = 1, compute_CSR = 1, transverse_on = 1, 
                    pbins = 20, obins = 30, plim = 5, olim = 5,
                    write_beam = None, write_wakes = True, write_name = ''):
    """ Initializes the CSR_params dictionary"""

    keys = ["workdir", "apply_CSR", "compute_CSR", "transverse_on", "pbins", "obins", "plim", "olim",
            "write_beam", "write_wakes", "write_name"]
    values = [full_path(workdir), apply_CSR, compute_CSR, transverse_on, pbins, obins, plim, olim,
            write_beam, write_wakes, write_name]

    CSR_params = {k: v for k, v in zip(keys, values)}

    return CSR_params

def init_snapshots(step_num, input_dict):
        """
        Initializes all snapshot instances which preallocates all np arrays
        """
        snapshots = [None]*step_num

        # For each step initialize a snapshot
        for step_index in range(step_num):
            snapshots[step_index] = Step_Snapshot(**input_dict.get("particle_deposition", {}))

        return snapshots
        
