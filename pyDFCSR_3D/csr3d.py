# Import standard library modules
import os

# Import third-party modules
import numpy as np
import matplotlib.pyplot as plt
import h5py
from bmadx import  Drift, SBend, Quadrupole, Sextupole

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
        self.dump_beam(directory="/Users/amf16/Desktop/SULI 2024/Simulation Output")
        
        # Starting at the second snapshot, we propagate, populate, compute CSR, and apply CSR
        for step_index, snapshot in enumerate(self.step_snapshots[1:], start=1):

            # Propagate the beam to the current step position
            self.beam.track(self.lattice.bmadx_elements[step_index-1])

            # Populate the step_snapshot with the beam distribution
            CSR_params, CSR_matrices, CSR_mesh = snapshot.populate(self.beam)

            # Compute the formation length of the beam at this moment in time
            formation_length = self.get_formation_length(step_index)
            print(formation_length)
            # Compute CSR_wake on the mesh
            dE_vals, x_kick_vals = self.compute_CSR_on_mesh(CSR_mesh, formation_length)

            # Apply to beam
            self.beam.apply_wakes(dE_vals, x_kick_vals, CSR_params, CSR_matrices, CSR_mesh, self.lattice.step_size)

            # Populate the step_snapshot with the new beam distribution, do not update the CSR_mesh
            snapshot.populate(self.beam, update_CSR_mesh = False)

            # Dump the beam at this step if desired by the user
            if (step_index+1) in self.CSR_mesh_params["write_beam"]:
                self.dump_beam(directory="/Users/amf16/Desktop/SULI 2024/Simulation Output")

            # Update the statistics dict
            self.update_statistics(step_index)
        
        # Dumps the final beam
        self.dump_beam(directory="/Users/amf16/Desktop/SULI 2024/Simulation Output")

    # TODO: fix the case for non dipole elements
    def get_formation_length(self, step_index):
        """
        Computes the formation length for the beam at inputed step referenece trajectory and current beam shape.
        Note that we use the formation length of the element in which the prvious step was in
        """
        # Index the previous element
        previous_bmadx_element = self.lattice.bmadx_elements[step_index-1][-1]

        # Check to see if it is a dipole or not
        if isinstance(previous_bmadx_element, SBend):
            R = 1/(previous_bmadx_element.G)
        
        else:
            R = 1.0

        return (24 * (R ** 2) * self.beam.sigma_z) ** (1 / 3)

    def compute_CSR_on_mesh(self, CSR_mesh, formation_length):
        """
        Computes the CSR wake at each point in the CSR_mesh
        Parameters:
            CSR_mesh: the mesh on which each vertex the CSR_wake is to be computed
            formation_length: the formation_length of the beam at this step
        """
        # Flatten the CSR_mesh
        Z = CSR_mesh[:,:,0].flatten()
        X = CSR_mesh[:,:,1].flatten()

        # Define the arrays where the wake values will be stored
        dE_vals = np.zeros(len(Z))
        x_kick_vals = np.zeros(len(Z))

        # Precompute these beam characteristic here
        linear_fit = self.beam.linear_fit
        x_mean = self.beam.mean_x
        x_std = self.beam.sigma_x
        z_std = self.beam.sigma_z

        # For each point on the CSR_mesh, compute dE and x_kick and update their respective arrays
        for index in range(len(X)):
            s = self.beam.position + Z[index]
            x = X[index]

            dE_vals[index], x_kick_vals[index] = self.compute_CSR_at_point(s, x, formation_length, linear_fit, x_mean, x_std, z_std)

        # Reshape dE_vals and x_kick_vals to be the dimension of the mesh again
        original_shape = CSR_mesh.shape[:2]
        dE_vals = dE_vals.reshape(original_shape)
        x_kick_vals = x_kick_vals.reshape(original_shape)

        return dE_vals, x_kick_vals

    def compute_CSR_at_point(self, s, x, formation_length, linear_fit, x_mean, x_std, z_std):
        """
        Helper function to compute_CSR_on_mesh, computes the CSR wake at a singular point on the CSR mesh
        Parameters:
            s and x: position of point on which we want to compute CSR wake
            linear_fit: a named tupled with slope and intercept
            beam_position: the position of the beam on the reference trajectory
        """
        # Compute the integration areas
        integration_areas = self.get_integration_areas(s, x, formation_length, linear_fit, x_mean, x_std, z_std)

        # Initialize the dE and x_kick
        dE = 0
        x_kick = 0

        # Compute the integrand over each area, integrate, and then sum the contribution
        for area in integration_areas:
            integrand_z, integrand_x = self.get_CSR_integrand(s, x, area)

            # Integrate over these real quick using trap
            dE += 0 # trap(integrand_z)
            x_kick += 0# trap(integrand_x)

        return dE, x_kick
    
    def get_integration_areas(self, s, x, formation_length, linear_fit, x_mean, x_std, z_std):
        """
        Helper function to compute_CSR_at_point. Given a point on the CSR mesh, computes the three or 
        four areas of space in the lab frame over which we must integrate
        Parameters:
            s and x: position of point on which we want to compute CSR wake
            formation_length: the formation length of the beam at the current step
            linear_fit: a named tupled with slope and intercept
            x_mean, x_std, z_std: beam distribution function characteristics
        Returns:
            integration_areas: an array containing [x_mesh, s_mesh] meshgrids for each integration area
        """
        x0 = (s - self.beam.position)*linear_fit.slope

        # Boolean tracking if the beam is sufficently tilted
        sufficent_tilt = np.abs(linear_fit.slope) > 1

        # If the band is sufficently tilted, then the integration region changes
        if sufficent_tilt:

            # Construct the integration regions in a way that depends on  which way the band is tiled
            if linear_fit.slope > 0:
                tan_alpha = -2 * linear_fit.slope / (1 - linear_fit.slope ** 2)  # alpha = pi - 2 theta, tan_alpha > 0
                d = (10 * x_std + x_mean - x) / tan_alpha

                s4 = s + 3 * z_std
                s3 = np.max((0, s - d))
                s2 = s3 - 200 * z_std

                # area 1
                x1_l = x + 0.1 * x_std
                x1_r = x + 10 * x_std

                # area 2
                x2_l = x - 3 * x_std
                x2_r = x1_l

                # area 3
                x3_l = x0 - 5 * x_std
                x3_r = x0 + 5 * x_std

                x4_l = x0 - 20 * x_std
                x4_r = x0 + 20 * x_std

            else:
                tan_alpha = 2 * linear_fit.slope / (1 - linear_fit.slope ** 2)
                d = -(x_mean - x - 10 * x_std) / tan_alpha

                s4 = s + 3 * z_std
                s3 = np.max((0, s - d))
                s2 = s3 - 200 * z_std

                # area 1
                x1_l = x - 10 * x_std
                x1_r = x - 1 * x_std

                # area 2
                x2_l = x1_r
                x2_r = x + 3 *x_std

                # area 3
                x3_l = x0 - 5 * x_std
                x3_r = x0 + 5 * x_std

                x4_l = x0 - 20 * x_std
                x4_r = x0 + 20 * x_std

            # Make sure that we cap the smallest intergration bound at s=0, where the nominal path begins
            s1 = np.max((0, s2 - self.integration_params["n_formation_length"] * formation_length))
            
            sp1 = np.linspace(s1, s2, self.integration_params["zbins"])
            sp2 = np.linspace(s2, s3, self.integration_params["zbins"])
            sp3 = np.linspace(s3, s4, self.integration_params["zbins"])
            xp1 = np.linspace(x1_l, x1_r, self.integration_params["xbins"])
            xp2 = np.linspace(x2_l, x2_r, self.integration_params["xbins"])
            xp3 = np.linspace(x3_l, x3_r, self.integration_params["xbins"])
            xp4 = np.linspace(x4_l, x4_r, 2*self.integration_params["xbins"])

            [xp_mesh1, sp_mesh1] = np.meshgrid(xp4, sp1, indexing='ij')
            [xp_mesh2, sp_mesh2] = np.meshgrid(xp3, sp2, indexing = 'ij')
            [xp_mesh3, sp_mesh3] = np.meshgrid(xp1, sp3, indexing='ij')
            [xp_mesh4, sp_mesh4] = np.meshgrid(xp2, sp3, indexing='ij')
            integration_areas = [[xp_mesh1, sp_mesh1], [xp_mesh2, sp_mesh2], [xp_mesh3, sp_mesh3], [xp_mesh4, sp_mesh4]]

        else:
            s2 = s - 500 * z_std
            s3 = s - 20 *z_std
            s4 = s + 5 * z_std

            x1_w = x0 - 20 * x_std
            x2_w = x0 + 20 * x_std

            x1_n = x0 - 10 * x_std
            x2_n = x0 + 10 * x_std

            # Make sure that we cap the smallest intergration bound at s=0, where the nominal path begins
            s1 = np.max((0, s2 - self.integration_params["n_formation_length"] * formation_length))

            # Initalize the integration meshes with the desired number of bins
            sp1 = np.linspace(s1, s2, self.integration_params["zbins"])
            sp2 = np.linspace(s2, s3, self.integration_params["zbins"])
            sp3 = np.linspace(s3, s4, self.integration_params["zbins"])
            xp_w = np.linspace(x1_w, x2_w, 2*self.integration_params["xbins"])
            xp_n = np.linspace(x1_n, x2_n, self.integration_params["xbins"])

            [xp_mesh1, sp_mesh1] = np.meshgrid(xp_w, sp1, indexing='ij')
            [xp_mesh2, sp_mesh2] = np.meshgrid(xp_n, sp2, indexing='ij')
            [xp_mesh3, sp_mesh3] = np.meshgrid(xp_n, sp3, indexing='ij')
            integration_areas = [[xp_mesh1, sp_mesh1], [xp_mesh2, sp_mesh2], [xp_mesh3, sp_mesh3]]

        return integration_areas
    
    def get_CSR_integrand(self, s, x, area):
        """
        Helper function to compute_CSR_at_point, finds the integrand contribution (W1 + W2 + W3) of the inputed integration area
        to the specific point
        """

        return 0, 0

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

        print("Beam at position {} is written to {}".format(self.beam.position, filename))

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
 


