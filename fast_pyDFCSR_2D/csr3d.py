# Import standard library modules
import os
import sys

# Import third-party modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import h5py
from bmadx import  Drift, SBend, Quadrupole, Sextupole
from tqdm import tqdm
from line_profiler import profile

# Import modules specific to this package
from utility_functions import check_input_consistency, isotime
from yaml_parser import parse_yaml
from attribute_initializer import init_attributes
from lattice import Lattice
from beam import Beam
from df_tracker import DF_Tracker
from interpolation import *

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
         self.df_tracker) = init_attributes(self.input_dict)

        # charge in C (8.98755e-6 MeV/m for 1nC/m^2)
        self.CSR_scaling = 8.98755e3 * self.beam.charge 

        # Get the current time
        self.timestamp = isotime()

        # Create the prefix (for naming)
        self.prefix = f'{self.CSR_mesh_params["write_name"]}-{self.timestamp}'

        # If we are using parallel computing
        if parallel:
            pass
        else:
            self.parallel = False

        print("done with init")

    def run(self):
        """
        Computes the CSR wake at each step in the lattice
        """
        
        # Populate the df_tracker with the first step
        self.df_tracker.populate_step(0, self.beam)
        self.update_statistics(0)
        
        # Starting at the second step, we propagate, populate, compute CSR, and apply CSR
        #for step_index in tqdm(range(1, self.df_tracker.total_steps)):
        for step_index in range(1, 2):
            # Propagate the beam to the current step position
            self.beam.track(self.lattice.bmadx_elements[step_index-1])

            # Populate the  with the beam distribution
            csr_params, t2_csr, R_inv_csr, CSR_mesh = self.df_tracker.populate_step(step_index, self.beam)

            # Compute the formation length of the beam at this moment in time
            formation_length = self.get_formation_length(step_index)

            # Compute CSR_wake on the mesh
            dE_vals, x_kick_vals = self.compute_CSR_on_mesh(CSR_mesh, formation_length, step_index)

            # Apply to beam
            self.beam.apply_wakes(dE_vals, x_kick_vals, csr_params, t2_csr, R_inv_csr, self.lattice.step_size)

            # Populate the df_tracker with the new beam distribution, do not update the CSR_mesh
            self.df_tracker.populate_step(step_index, self.beam, update_CSR_mesh = False)

            # Dump the beam at this step if desired by the user
            if (step_index+1) in self.CSR_mesh_params["write_beam"]:
                self.dump_beam(directory="/Users/amf16/Desktop/SULI 2024/Simulation Output")

            # Update the statistics dict
            self.update_statistics(step_index)

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

    def compute_CSR_on_mesh(self, CSR_mesh, formation_length, step_index):
        """
        Computes the CSR wake at each point in the CSR_mesh
        Parameters:
            CSR_mesh: the mesh on which each vertex the CSR_wake is to be computed
            formation_length: the formation_length of the beam at this step
        """
        # Flatten the CSR_mesh
        Z = CSR_mesh[:,:,1].flatten()
        X = CSR_mesh[:,:,0].flatten()

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

            dE_vals[index], x_kick_vals[index] = self.compute_CSR_at_point(s, x, formation_length, linear_fit, x_mean, x_std, z_std, step_index)

        # Reshape dE_vals and x_kick_vals to be the dimension of the mesh again
        original_shape = CSR_mesh.shape[:2]
        dE_vals = dE_vals.reshape(original_shape)
        x_kick_vals = x_kick_vals.reshape(original_shape)

        return dE_vals, x_kick_vals

    def compute_CSR_at_point(self, s, x, formation_length, linear_fit, x_mean, x_std, z_std, step_index, plot=False):
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
            integrand_z, integrand_x = self.get_CSR_integrand(s, x, area, step_index)

            # Integrate over these real quick using trap
            dE += -self.CSR_scaling * np.trapz(y=np.trapz(y=integrand_z, x=area[0][:, 0], axis=0), x=area[1][0,:])
            x_kick += self.CSR_scaling * np.trapz(y=np.trapz(y=integrand_x, x=area[0][:, 0], axis=0), x=area[1][0,:])

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
    
    #@profile
    def get_CSR_integrand(self, s, x, area, step_index):
        """
        Helper function to compute_CSR_at_point, finds the integrand contribution (W1 + W2 + W3) of the inputed integration area
        to the specific point
        """
        # Define t
        t = self.beam.position

        # Flatten the meshgrid
        sp_flat = area[1].ravel()
        xp_flat = area[0].ravel()

        # Get x & x prime and y & y prime in the lab frame
        # the 'prime' variables are the integration mesh grid converted to lab frame coordinates
        X0_s = interpolate1D(xval = np.array([s]), data = self.lattice.lab_frame_sample_coords[:, 0], min_x = self.lattice.min_s,
                             delta_x = self.lattice.delta_s)[0]
        X0_sp = interpolate1D(xval = sp_flat, data = self.lattice.lab_frame_sample_coords[:, 0], min_x = self.lattice.min_s,
                              delta_x = self.lattice.delta_s)
        Y0_s = interpolate1D(xval = np.array([s]), data = self.lattice.lab_frame_sample_coords[:, 1], min_x = self.lattice.min_s,
                             delta_x = self.lattice.delta_s)[0]
        Y0_sp = interpolate1D(xval = sp_flat, data = self.lattice.lab_frame_sample_coords[:, 1], min_x = self.lattice.min_s,
                              delta_x = self.lattice.delta_s)

        # Do the same for the normal and tangential vectors
        n_vec_s_x = interpolate1D(xval = np.array([s]), data = self.lattice.sample_n_vecs[:, 0], min_x = self.lattice.min_s,
                                  delta_x = self.lattice.delta_s)[0]
        n_vec_sp_x =interpolate1D(xval = sp_flat, data = self.lattice.sample_n_vecs[:, 0], min_x = self.lattice.min_s,
                                  delta_x = self.lattice.delta_s)
        n_vec_s_y = interpolate1D(xval=np.array([s]), data=self.lattice.sample_n_vecs[:, 1], min_x=self.lattice.min_s,
                                  delta_x=self.lattice.delta_s)[0]
        n_vec_sp_y = interpolate1D(xval=sp_flat, data=self.lattice.sample_n_vecs[:, 1], min_x=self.lattice.min_s,
                                   delta_x=self.lattice.delta_s)
        tau_vec_s_x = interpolate1D(xval=np.array([s]), data=self.lattice.sample_tau_vecs[:, 0], min_x=self.lattice.min_s,
                                  delta_x=self.lattice.delta_s)[0]
        tau_vec_sp_x = interpolate1D(xval=sp_flat, data=self.lattice.sample_tau_vecs[:, 0], min_x=self.lattice.min_s,
                                   delta_x=self.lattice.delta_s)
        tau_vec_s_y = interpolate1D(xval=np.array([s]), data=self.lattice.sample_tau_vecs[:, 1], min_x=self.lattice.min_s,
                                  delta_x=self.lattice.delta_s)[0]
        tau_vec_sp_y = interpolate1D(xval=sp_flat, data=self.lattice.sample_tau_vecs[:, 1], min_x=self.lattice.min_s,
                                   delta_x=self.lattice.delta_s)

        # Comupte the magnitude of the different between r and rp (r prime) vectors
        r_minus_rp_x = X0_s - X0_sp + x * n_vec_s_x - xp_flat * n_vec_sp_x
        r_minus_rp_y = Y0_s - Y0_sp + x * n_vec_s_y - xp_flat * n_vec_sp_y
        r_minus_rp = np.sqrt(r_minus_rp_x**2 + r_minus_rp_y**2)

        # Retarded time array for each point in meshgrid
        t_ret = t - r_minus_rp

        # The effective radius from the lab frame origin for each lattice step
        rho_sp = np.zeros(sp_flat.shape)

        # Populate rho_sp
        for count in range(self.lattice.element_num):
            if count == 0:
                rho_sp[sp_flat < self.lattice.element_distances[count]] = self.lattice.element_rho_vals[count]
            else:
                rho_sp[(sp_flat < self.lattice.element_distances[count]) & (sp_flat >= self.lattice.element_distances[count - 1])] = self.lattice.element_rho_vals[count]

        ranges = self.lattice.step_ranges[0:step_index]
        # Interpolate the velocity of the beam at the current point
        translated_point = translate_points(np.array([t]), np.array([s-t]), np.array([x]),
                                             self.df_tracker.t1_h, self.df_tracker.C_inv_h, self.df_tracker.R_inv_h, self.df_tracker.t2_h,
                                             self.lattice.step_ranges, np.array([0]), np.zeros((1, 5)))
        vx = interpolate3D(translated_point, [self.df_tracker.beta_x], self.lattice.step_size, np.zeros((5, 1)))[0][0]

        # Initialize translated mesh points and the index of each point
        translated_points = np.zeros((len(t_ret), 5))
        p_indices = np.arange(len(t_ret), dtype=int)

        # Populate the translated mesh points with all the points on the mesh
        translated_points = translate_points(t_ret, sp_flat - t_ret, xp_flat,
                                             self.df_tracker.t1_h, self.df_tracker.C_inv_h, self.df_tracker.R_inv_h, self.df_tracker.t2_h,
                                             self.lattice.step_ranges, p_indices, translated_points)

        # Interpolate retarted time quantites!
        interp_result = np.zeros((5, len(t_ret)), dtype=np.float64)
        interp_data = (self.df_tracker.densities, 
                       self.df_tracker.beta_x, 
                       self.df_tracker.partial_density_x, 
                       self.df_tracker.partial_density_z,
                       self.df_tracker.partial_beta_x)
        
        interp_result = interpolate3D(translated_points, interp_data, self.lattice.step_size, interp_result)

        # Unpack the interpolate result
        density_ret = interp_result[0]
        beta_x_ret = interp_result[1]
        partial_density_x_ret = interp_result[2]
        partial_density_z_ret = interp_result[3]
        partial_beta_x = interp_result[4]

        # TODO: More accurate vx, maybe add vs
        # TODO: Compute vx
        vs = 1
        vs_ret = 1
        vs_s_ret = 0

        # Accounts for transfer to the lab frame
        scale_term =  1 + xp_flat*rho_sp

        # Compute velocity in the lab frame for current time and retarded
        velocity_x = vs * tau_vec_s_x + vx * n_vec_s_x
        velocity_y = vs * tau_vec_s_y + vx * n_vec_s_y
        velocity_ret_x = vs_ret * tau_vec_sp_x + beta_x_ret * n_vec_sp_x
        velocity_ret_y = vs_ret * tau_vec_sp_y + beta_x_ret * n_vec_sp_y

        nabla_density_ret_x = partial_density_x_ret  * n_vec_sp_x + partial_density_z_ret / scale_term * tau_vec_sp_x
        nabla_density_ret_y = partial_density_x_ret * n_vec_sp_y + partial_density_z_ret / scale_term * tau_vec_sp_y

        div_velocity = vs_s_ret + partial_beta_x

        # TODO: Consider using general form
        ## general form
        # part1: beta dot beta prime
        part1 = velocity_x * velocity_ret_x + velocity_y * velocity_ret_y

        # Some numerators found in the longitudinal wake integrals
        CSR_numerator1 = scale_term * ((velocity_x - part1 * velocity_ret_x) * nabla_density_ret_x  + \
                          (velocity_y - part1 * velocity_ret_y)*nabla_density_ret_y)
        CSR_numerator2 = -scale_term * part1 * density_ret * div_velocity

        CSR_integrand_z = CSR_numerator1 /r_minus_rp + (CSR_numerator2) / r_minus_rp

        n_minus_np_x = n_vec_s_x - n_vec_sp_x
        n_minus_np_y = n_vec_s_y - n_vec_sp_y

        # part1: (r-r')(n - n')
        part1 = r_minus_rp_x * n_minus_np_x + r_minus_rp_y * n_minus_np_y

        #part2: n tau'
        part2 = n_vec_s_x * tau_vec_sp_x + n_vec_s_y * tau_vec_sp_y

        # part3: partial density/partial t_ret
        partial_density = - (velocity_ret_x * nabla_density_ret_x + velocity_ret_y * nabla_density_ret_y) - \
                          density_ret * div_velocity

        # Three integrands for logitudinal wake
        W1 = scale_term * part1 / (r_minus_rp * r_minus_rp * r_minus_rp) * density_ret
        W2 = scale_term * part1 / (r_minus_rp * r_minus_rp) * partial_density
        W3 = -scale_term * part2 / r_minus_rp * partial_density

        CSR_integrand_x = W1 + W2 + W3
        CSR_integrand_x = CSR_integrand_x.reshape(area[0].shape)
        CSR_integrand_z = CSR_integrand_z.reshape(area[1].shape)

        return CSR_integrand_z, CSR_integrand_x

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
            """
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
            """

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

    def plot_all_meshes_on_reference_traj(self, step_index, integration_areas, s, x):
        """
        Plots the histogram mesh, CSR mesh, and three or four integration areas
        """
        fig, ax = plt.subplots()

        # histogram mesh
        h_mesh = self.step_snapshots[step_index].h_coords
        h_mesh_corners = [[h_mesh[-1][0][1], h_mesh[-1][0][0]], [h_mesh[0][0][1], h_mesh[0][0][0]], [h_mesh[0][-1][1], h_mesh[0][-1][0]], [h_mesh[-1][-1][1], h_mesh[-1][-1][0]]]

        # CSR mesh
        CSR_mesh = self.step_snapshots[step_index].CSR_coords
        CSR_mesh_corners = [[CSR_mesh[-1][0][1], CSR_mesh[-1][0][0]], [CSR_mesh[0][0][1], CSR_mesh[0][0][0]], [CSR_mesh[0][-1][1], CSR_mesh[0][-1][0]], [CSR_mesh[-1][-1][1], CSR_mesh[-1][-1][0]]]

        h_polygon = Polygon(h_mesh_corners, closed=True, edgecolor="black", facecolor="cyan", alpha=0.5, label="DF histogram mesh")
        CSR_polygon = Polygon(CSR_mesh_corners, closed=True, edgecolor="black", facecolor="red", alpha=0.5, label="CSR mesh")

        ax.add_patch(h_polygon)
        ax.add_patch(CSR_polygon)

        colors = ["blue", "green", "yellow", "purple"]
        for i, area in enumerate(integration_areas):
            x_mesh = area[0]
            s_mesh = area[1] - self.beam.position
            polygon_corners = [[s_mesh[-1][0], x_mesh[-1][0]], [s_mesh[0][0], x_mesh[0][0]], [s_mesh[0][-1], x_mesh[0][-1]], [s_mesh[-1][-1], x_mesh[-1][-1]]]
            polygon = Polygon(polygon_corners, edgecolor="black", facecolor=colors[i], alpha=0.5, label="integration mesh #"+str(i+1))
            ax.add_patch(polygon)

        #self.step_snapshots[step_index].plot_mesh(ax, CSR_mesh[:, :, 1], CSR_mesh[:, :, 0])
        
        # Add the point on the CSR mesh which the integration areas are for
        ax.scatter(s-self.beam.position, x, color="black", label="point to compute wake")

        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        ax.set_title("All meshes")
        #ax.set_aspect('equal')
        ax.legend()
        plt.show()

    def demonstrate_coordinate_transfer(self):
        """
        Verifies that the index and position transfer works
        """
        # populate the first snapshot
        self.df_tracker.populate_step(0, self.beam)

        # pick some random positions on the beam to compute indices
        # "0" is the retarded time
        test_positions = np.array([[0, self.beam.z[5], self.beam.x[5]],
                          [0, self.beam.z[10], self.beam.x[10]],
                          [0, self.beam.z[25], self.beam.x[25]],
                          [0, self.beam.z[1], self.beam.x[1]],
                          [0, self.beam.z[2], self.beam.x[2]]])
        
        # fake retarded time ranges
        ranges = [[0,1]]
        p_indices = np.arange(5, dtype=int)
        translated_points = np.zeros((5, 5))

        translated_points = translate_points(test_positions[:, 0], test_positions[:,1], test_positions[:,2],
                                             self.df_tracker.t1_h, self.df_tracker.C_inv_h, self.df_tracker.R_inv_h, self.df_tracker.t2_h,
                                             ranges, p_indices, translated_points)



        # Make the plot for the beam
        fig, ax = plt.subplots()

        # Plot the histogram mesh
        X = self.df_tracker.h_coords[0][:, :, 0]
        Z = self.df_tracker.h_coords[0][:, :, 1]
        self.plot_mesh(ax, Z, X, color="black")

        # Plot the test points
        for point in test_positions:
            ax.scatter(point[1], point[2], color="red", s=10)

        ax.axis("equal")
        plt.show()

        # Make the plot for the coordinate space
        fig, ax = plt.subplots()

        # Plot the coordinate mesh
        self.plot_mesh(ax, Z, X, color="blue")

        # Make a coordinate mesh and plot it
        mesh_z = np.arange(self.df_tracker.h_params["pbins"])
        mesh_x = np.arange(self.df_tracker.h_params["obins"])
        Z, X = np.meshgrid(mesh_z, mesh_x)
        self.plot_mesh(ax, Z, X, "blue")

        # Plot the test_positions in coordinate space
        for point in translated_points:
            ax.scatter(point[3], point[4], marker="x", color="black", s=20)


        ax.axis("equal")
        plt.show()


    def plot_mesh(self, ax, Z, X, color='black'):
        """
        Plots the mesh
        """
        # Plot vertical grid lines (lines along X)
        for i in range(X.shape[1]):
            ax.plot(Z[:, i], X[:, i], color, linewidth=1.0)

        # Plot horizontal grid lines (lines along Y)
        for j in range(X.shape[0]):
            ax.plot(Z[j, :], X[j, :], color, linewidth=1.0)
 


