# Import standard library modules
import os

# Import third-party modules
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Import modules specific to this package
from utility_functions import check_input_consistency, isotime
from yaml_parser import parse_yaml
from attribute_initializer import init_attributes
from lattice import Lattice

"""
Module Name: csr3d.py

Contains the CSR3D class which is the controller class for CSR wake computations
"""

class CSR3D:
    """
    Class to compute 3D CSR wake
    """
    def __init__(self, input_file, parallel = False):
        """
        Creates an instance of the CSR3D class
        Parameters:
            input_file: assumed to be a configuration in YMAL file format that can be converted to dict
            parallel: boolean, indicate if parallel processing should be used for computations
        Returns:
            instance of CSR3D
        """
        # Convert YAML data into input dictionary
        self.input_dict = parse_yaml(input_file)

        # Check that input file is in the correct format
        check_input_consistency(self.input_dict)

        # Create the statistics and parameters dictionary along with the 3 main class instances
        (self.statistics,
         self.integration_params,
         self.CSR_mesh_params,
         self.lattice,
         self.beam,
         self.step_snapshots) = init_attributes(self.input_dict)

        # Get the current time
        self.timestamp = isotime()

        # Create the prefix (for naming)
        self.prefix = f'{self.CSR_mesh_params["write_name"]}-{self.timestamp}'

        # If we are using parallel computing
        if parallel:
            pass
        else:
            self.parallel = False

    def run(self):
        """
        Computes the CSR wake at each step in the lattice
        """
        
        # Populate the first snapshot
        self.step_snapshots[0].populate(self.beam)
        self.update_statistics(0)
        self.dump_beam(directory="/Users/treyfischbach/Desktop/Stuff/Research/SLAC 2024/Program Output/Benchmarking new code to old code")
        
        # Starting at the second snapshot, we propagate, populate, compute CSR, and apply CSR
        for step_index, snapshot in enumerate(self.step_snapshots[1:], start=1):

            # Propagate the beam to the current step position
            self.beam.track(self.lattice.bmadx_elements[step_index-1])

            # Populate the step_snapshot with the beam distribution
            CSR_params, CSR_matrices, CSR_mesh = snapshot.populate(self.beam)

            # Compute CSR_wake on the mesh
            dE_vals, x_kick_vals = self.compute_CSR_on_mesh(CSR_mesh)

            # Apply to beam
            self.beam.apply_wakes(dE_vals, x_kick_vals, CSR_params, CSR_matrices, CSR_mesh, self.lattice.step_size)

            # Populate the step_snapshot with the new beam distribution, do not update the CSR_mesh
            snapshot.populate(self.beam, update_CSR_mesh = False)

            # Dump the beam at this step if desired by the user
            if (step_index+1) in self.CSR_mesh_params["write_beam"]:
                self.dump_beam(directory="/Users/treyfischbach/Desktop/Stuff/Research/SLAC 2024/Program Output/Benchmarking new code to old code")

            # Update the statistics dict
            self.update_statistics(step_index)
        
        # Dumps the final beam
        self.dump_beam(directory="/Users/treyfischbach/Desktop/Stuff/Research/SLAC 2024/Program Output/Benchmarking new code to old code")

    def compute_CSR_on_mesh(self, CSR_mesh):
        """
        Computes the CSR wake at each point in the CSR_mesh
        """
        # Flatten the CSR_mesh
        Z = CSR_mesh[:,:,0].flatten()
        X = CSR_mesh[:,:,1].flatten()

        # Define the arrays where the wake values will be stored
        dE_vals = np.zeros(len(Z))
        x_kick_vals = np.zeros(len(Z))

        # For each point on the CSR_mesh, compute dE and x_kick and update their respective arrays
        for index in range(len(X)):
            s = self.beam.s_position + Z[index]
            x = X[index]

            dE_vals[index], x_kick_vals[index] = self.compute_CSR_at_point(s, x)

        # Reshape dE_vals and x_kick_vals to be the dimension of the mesh again
        original_shape = CSR_mesh.shape[:2]
        dE_vals = dE_vals.reshape(original_shape)
        x_kick_vals = x_kick_vals.reshape(original_shape)

        return dE_vals, x_kick_vals

    def compute_CSR_at_point(self, s, x):
        """
        Helper function to compute_CSR_on_mesh, computes the CSR wake at a singular point on the CSR mesh
        """
        # Compute the integration areas
        integration_areas = self.get_integration_areas(s, x)

        # Initialize the dE and x_kick
        dE = 0
        x_kick = 0

        # Compute the integrand over each area, integrate, and then sum the contribution
        for area in integration_areas:
            integrand_z, integrand_x = self.get_CSR_integrand(area)

            # Integrate over these real quick using trap
            dE += 0 # trap(integrand_z)
            x_kick += 0# trap(integrand_x)

        return dE, x_kick
    
    def get_integration_areas(self, s, x):
        """
        Helper function to compute_CSR_at_point. Given a point on the CSR mesh, computes the three or 
        four areas of space in the lab frame over which we must integrate
        """
        integration_areas = []

        return integration_areas
    
    def get_CSR_integrand(self, s, x, t, integration_area):
        """
        Helper function to compute_CSR_at_point, finds the integrand contribution (W1 + W2 + W3) of the inputed integration area
        to the specific point
        """
        return 0

    def update_statistics(self, step):
        """
        Updates the statistics dictionary with the current step's beam characteristics
        """
        twiss = self.beam.twiss
        self.statistics['twiss']['alpha_x'][step] = twiss['alpha_x']
        self.statistics['twiss']['beta_x'][step] = twiss['beta_x']
        self.statistics['twiss']['gamma_x'][step] = twiss['gamma_x']
        self.statistics['twiss']['emit_x'][step] = twiss['emit_x']
        self.statistics['twiss']['eta_x'][step] = twiss['eta_x']
        self.statistics['twiss']['etap_x'][step] = twiss['etap_x']
        self.statistics['twiss']['norm_emit_x'][step] = twiss['norm_emit_x']
        self.statistics['twiss']['alpha_y'][step] = twiss['alpha_y']
        self.statistics['twiss']['beta_y'][step] = twiss['beta_y']
        self.statistics['twiss']['gamma_y'][step] = twiss['gamma_y']
        self.statistics['twiss']['emit_y'][step] = twiss['emit_y']
        self.statistics['twiss']['eta_y'][step] = twiss['eta_y']
        self.statistics['twiss']['etap_y'][step] = twiss['etap_y']
        self.statistics['twiss']['norm_emit_y'][step] = twiss['norm_emit_y']
        self.statistics['slope'][step, :] = self.beam._slope
        self.statistics['sigma_x'][step] = self.beam._sigma_x
        self.statistics['sigma_z'][step] = self.beam._sigma_z
        self.statistics['sigma_energy'][step] = self.beam.sigma_energy
        self.statistics['mean_x'][step] = self.beam._mean_x
        self.statistics['mean_z'][step] = self.beam._mean_z
        self.statistics['mean_energy'][step] = self.beam.mean_energy

    def save_data(self, directory="", filename=""):
        """
        Once all step_snapshots have been populated we save their data into an h5 file
        """
        if not filename:
            filename = f"{self.prefix}.h5"
    
        if not directory:
            directory = '.'  # Default to current directory if no directory is specified
        
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Save snapshot instances and statistics dictionary to an HDF5 file
        with h5py.File(file_path, 'w') as f:
            # snapshots
            for i, instance in enumerate(self.step_snapshots):
                grp = f.create_group(f'step_snapshot_{i}')
                grp.create_dataset('h_coords', data=instance.h_coords)
                grp.create_dataset('density', data=instance.density)
                grp.create_dataset('partial_density_x', data=instance.partial_density_x)
                grp.create_dataset('partial_density_z', data=instance.partial_density_z)
                grp.create_dataset('beta_x', data=instance.beta_x)
                grp.create_dataset('partial_beta_x', data=instance.partial_beta_x)
                grp.attrs['x_mean'] = instance.x_mean
                grp.attrs['z_mean'] = instance.z_mean
                grp.attrs['tilt_angle'] = instance.tilt_angle
                grp.attrs['s_val'] = instance.s_val

            # statistics
            stats_grp = f.create_group('statistics')
            for key, value in self.statistics.items():
                if isinstance(value, dict):
                    sub_grp = stats_grp.create_group(key)
                    for sub_key, sub_value in value.items():
                        sub_grp.create_dataset(sub_key, data=sub_value)
                else:
                    stats_grp.create_dataset(key, data=value)
            
    def dump_beam(self, directory="", filename=""):
        """
        Record the current beam in the particle_group format
        """
        if not filename:
            filename = str(self.beam.step) + ".h5"
    
        if not directory:
            directory = '.'  # Default to current directory if no directory is specified
        
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        if os.path.isfile(filename):
            os.remove(filename)
            print("Existing file " + filename + " deleted.")

        print("Beam at position {} is written to {}".format(self.beam.s_position, filename))

        self.beam.particle_group.write(file_path)

    def demonstrate_coordinate_transfer(self):
        """
        Verifies that the index and position transfer works
        """
        # Make the plot
        fig, ax = plt.subplots()

        # populate the first snapshot
        self.df_tracker.snapshots[0].populate(self.beam)

        # test if index to position works
        test_indices = [np.array([0,0], dtype=np.float64),
                        np.array([3,0], dtype=np.float64),
                        np.array([0,2], dtype=np.float64),
                        np.array([8,10], dtype=np.float64),
                        np.array([2,11], dtype=np.float64),
                        np.array([19,19], dtype=np.float64)]

        for index in test_indices:
            position = self.df_tracker.snapshots[0].index2position(index)
            ax.scatter(position[0], position[1], color="purple", s=20)

        # test if position to index works
        test_positions = [np.array([0.0, 0.0], dtype=np.float64),
                          np.array([0.00011, 0.00005], dtype=np.float64),
                          np.array([0.00011, -0.00005], dtype=np.float64),
                          np.array([-0.00011, 0.00005], dtype=np.float64),
                          np.array([-0.00011, -0.00005], dtype=np.float64),
                          np.array([0.00015, -0.000025], dtype=np.float64)]
        
        # Plot the four nearest mesh vertices with the position marked as "x"
        for position in test_positions:
            indices = self.df_tracker.snapshots[0].position2index(position)
            ax.scatter(position[0], position[1], marker="x", color="black", s=20)

            for index in indices:
                index_position = self.df_tracker.snapshots[0].index2position(index) 
                ax.scatter(index_position[0], index_position[1], color="green", s=20)
        
        self.df_tracker.snapshots[0].plot_grid_transformation(self.beam, fig=fig, ax=ax)
 


