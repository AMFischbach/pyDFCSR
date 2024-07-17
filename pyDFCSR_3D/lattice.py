import numpy as np

from create_reference_traj import *
from yaml_parser import parse_yaml

"""
Module Name: lattice.py

Contains the Lattice class
"""

class Lattice():
    """
    lattice class to read lattice file and get characteristics such as reference trajectory
    maybe install a pointer for the position of the current beam
    """

    def __init__(self, input_lattice, init_energy):
        """
        Parameters:
            input_lattice: a one element dictionary with key 'lattice_input_file' and value with pathname.yaml (string)
        Returns:
            instance of Lattice
        """

        # Make sure that the input_lattice has the input_file path name
        assert 'lattice_input_file' in input_lattice, 'Error in parsing lattice: must include the keyword <lattice_input_file>'
        self.lattice_input_file = input_lattice['lattice_input_file']

        # Create lattice dictionary from pathname.yaml
        self.lattice_config = parse_yaml(self.lattice_input_file)

        # Verify that all necessary parameters are present in settings dictionary
        self.check_input(self.lattice_config)

        # TODO for monday: make one big "lattice" dictionary
        # Loops through all the elements and compiles their characteristics into arrays
        (self.element_distances,
         self.element_rho_vals,
         self.CSR_step_seperation) = get_element_characteristics(self.lattice_config)
        
        # Loops through all steps and compute their characteristics
        (self.step_size,
         self.total_steps,
         self.step_position,
         self.compute_CSR) = get_step_characteristics(self.lattice_config, self.element_distances, self.CSR_step_seperation)

        # Loops through the steps again to compute bmadx_elements for each step
        self.bmadx_elements = get_bmdax_elements(self.lattice_config, self.element_distances, self.step_position, init_energy)

        # Finely samples the reference trajectory at many points
        (self.sample_s_vals,
         self.lab_frame_sample_coords,
         self.sample_n_vecs,
         self.sample_tau_vecs) = get_trajectory_characteristics(self.lattice_config, self.element_distances)

        # Computes min, max, and average change of s along reference trajectory
        #self.min_s, self.max_s = self.step_s_vals[0], self.step_s_vals[-1]
        #self.delta_s = (self.max_s - self.min_s) / (self.step_s_vals.shape[0] - 1)
 
    def check_input(self, input):
        # Todo: check input for lattice
        assert 'step_size' in input, f'Required input parameter step_size to {self.__class__.__name__}.__init__(**kwargs) was not found.'

    @property
    def lattice_length(self):
        return self.element_distances[-1]