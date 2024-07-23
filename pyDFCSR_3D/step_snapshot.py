import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm
from line_profiler import profile

from beam import Beam
from histogram_functions import histogram_cic_2d

"""
Module Name: step_snapshot.py

Contains the Step_Snapshot class 
"""

class Step_Snapshot():

    def __init__(self, step_index, histo_mesh_params, CSR_mesh_params):
        """
        Initalizes a Step_Snapshot
        Parameters:
            total_steps: the total number of steps in the lattice
            histo_mesh_params: the parameters detailing the characteristics of the histogram meshes
            CSR_mesh_params: the parameters detailing the characteristics of the CSR_mesh_params
        """
        # Histogram mesh parameters
        self.h_params = histo_mesh_params

        # CSR mesh parameters
        self.CSR_params = CSR_mesh_params

        # Positions of each vertex of the histogram mesh
        self.h_coords = np.empty((self.h_params["obins"], self.h_params["pbins"], 2), dtype=np.float64)

        # Positions of each vertex of the CSR mesh
        self.CSR_coords = np.empty((self.CSR_params["obins"], self.CSR_params["pbins"], 2), dtype=np.float64)

        # Characteristics of beam distribution at this snapshot
        self.x_mean = None
        self.z_mean = None
        self.p_sd = None
        self.o_sd = None
        self.tilt_angle = None  # angle of the line of best fit from the horizontal axis

        # The s value of the step on the reference trajectory and the step index
        self.s_val = None
        self.step_index = step_index

        # Initialize matrices describing the histogram mesh transformation from the unit mesh
        self.h_matrices = {}
        self.h_matrices["t1"] = np.empty(2, dtype=np.float64)             # 1st tranform (in unit mesh)
        self.h_matrices["C"] = np.empty((2,2), dtype=np.float64)          # Compression
        self.h_matrices["C_inv"] = np.empty((2,2), dtype=np.float64)      # Compression inversion
        self.h_matrices["R"] = np.empty((2,2), dtype=np.float64)          # Rotation
        self.h_matrices["R_inv"] = np.empty((2,2), dtype=np.float64)      # Rotation inversion
        self.h_matrices["t2"] = np.empty(2, dtype=np.float64)             # 2nd tranform (to beam center)

        # Preallocate space for the numpy arrays (histograms)
        self.density = np.empty((self.h_params["obins"], self.h_params["pbins"]), dtype=np.float64)            # Density
        self.partial_density_x = np.empty((self.h_params["obins"], self.h_params["pbins"]), dtype=np.float64)  # Partial Derivative of density wrt x
        self.partial_density_z = np.empty((self.h_params["obins"], self.h_params["pbins"]), dtype=np.float64)  # Partial Derivative of density wrt x
        self.beta_x = np.empty((self.h_params["obins"], self.h_params["pbins"]), dtype=np.float64)             # x component of beta = v/c
        self.partial_beta_x = np.empty((self.h_params["obins"], self.h_params["pbins"]), dtype=np.float64)     # x component of partial beta

        # Initialize matrices describing the CSR mesh transformation from the unit mesh
        self.CSR_matrices = {}
        self.CSR_matrices["t1"] = np.empty(2, dtype=np.float64)             # 1st tranform (in unit mesh)
        self.CSR_matrices["C"] = np.empty((2,2), dtype=np.float64)          # Compression
        self.CSR_matrices["C_inv"] = np.empty((2,2), dtype=np.float64)      # Compression inversion
        self.CSR_matrices["R"] = np.empty((2,2), dtype=np.float64)          # Rotation
        self.CSR_matrices["R_inv"] = np.empty((2,2), dtype=np.float64)      # Rotation inversion
        self.CSR_matrices["t2"] = np.empty(2, dtype=np.float64)             # 2nd tranform (to beam center)

        # Maybe initialize matrices describing the CSR wake

    def populate(self, beam, update_CSR_mesh = True):
        """
        Converts a Beam object into a fully populated Step_Snapshot object
        Parameters:
            beam: Beam object to be used
            update_CSR_mesh: when we re update the stepsnapshot after the CSR wake has already
                             been applied to the beam, we do not want to update the CSR_mesh on
                             which the wake was computed.
        """
        # Populate beam characteristics
        self.x_mean = beam.mean_x
        self.z_mean = beam.mean_z
        self.p_sd, self.o_sd = beam.get_std_wrt_linear_fit()
        self.tilt_angle = beam.tilt_angle
        self.s_val = beam.position

        # Popultate transformation matrices for histogram mesh (not in ij format)
        (self.h_matrices["t1"],
        self.h_matrices["C"],
        self.h_matrices["C_inv"],
        self.h_matrices["R"],
        self.h_matrices["R_inv"],
        self.h_matrices["t2"]) = self.get_mesh_matrices(self.h_params) 

        # Populate histogram mesh coordinates
        self.h_coords = self.get_mesh_coordinates(self.h_params, self.h_matrices)

        # Populate density and beta_x histograms
        (self.density, 
         self.beta_x,
         self.partial_density_x,
         self.partial_density_z,
         self.partial_beta_x) = self.populate_2D_histograms(beam.z, beam.x, beam.px, self.h_params, self.h_matrices)

        # If we update the CSR_mesh we return it for quick computation
        if update_CSR_mesh:
            # Popultate transformation matrices for CSR mesh (not in ij format)
            (self.CSR_matrices["t1"],
            self.CSR_matrices["C"],
            self.CSR_matrices["C_inv"],
            self.CSR_matrices["R"],
            self.CSR_matrices["R_inv"],
            self.CSR_matrices["t2"]) = self.get_mesh_matrices(self.CSR_params) 

            # Populate CSR mesh coordinates
            self.CSR_coords = self.get_mesh_coordinates(self.CSR_params, self.CSR_matrices)

            return self.CSR_params, self.CSR_matrices, self.CSR_coords
        
    def get_mesh_matrices(self, mesh_params):
        """
        Given the mesh dimensions, creates the matrices which transform the mesh from coordinate space to cover the beam in z,x space
        """
        pbins, obins, plim, olim = mesh_params["pbins"], mesh_params["obins"], mesh_params["plim"], mesh_params["olim"]

        # Popultate transformation matrices (not in ij format)
        t1 = np.array([(pbins-1)/2, (obins-1)/2], dtype = np.float64) 
        C = np.array([[2*(plim * self.p_sd)/(pbins-1), 0.0],
                          [0.0, 2*(olim * self.o_sd)/(obins-1)]], dtype = np.float64)
        C_inv = np.linalg.inv(C)
        R = np.array([[np.cos(self.tilt_angle), -np.sin(self.tilt_angle)],
                          [np.sin(self.tilt_angle), np.cos(self.tilt_angle)]], dtype=np.float64)
        R_inv = np.linalg.inv(R)
        t2 = np.array([self.z_mean,self.x_mean], dtype=np.float64)

        return t1, C, C_inv, R, R_inv, t2

    def get_mesh_coordinates(self, mesh_params, mesh_matrices):
        """
        Creates a rotated and compressed mesh to match what ever the beam distribution looks like at this step
        Helper function to populate, populates the mesh coordinates. Also is used to create
        the mesh upon which the CSR wake is computed
        Parameters:
            mesh_coordinates: the mesh_coordinates attribute to be populated
        """
        pbins, obins= mesh_params["pbins"], mesh_params["obins"]
        t1, C, R, t2 = mesh_matrices["t1"], mesh_matrices["C"], mesh_matrices["R"], mesh_matrices["t2"]

        # Seed the mesh
        mesh_z = np.arange(pbins)
        mesh_x = np.arange(obins)

        Z, X = np.meshgrid(mesh_z, mesh_x)
        mesh_coords_list = np.stack((Z.flatten(), X.flatten()), axis=0).T

        # Shift mesh to center
        mesh_coords_list = mesh_coords_list - t1
        
        # Compress mesh
        mesh_coords_list = mesh_coords_list @ C.T

        # Rotate mesh
        mesh_coords_list = mesh_coords_list @ R.T

        # Translate mesh to center of beam distribution
        mesh_coords_list = mesh_coords_list + t2

        # Populate the mesh coordinates in ij format
        mesh_coords = np.empty((obins, pbins, 2), dtype=np.float64)
        mesh_coords[:,:,0] = mesh_coords_list[:, 1].reshape(obins, pbins)
        mesh_coords[:,:,1] = mesh_coords_list[:, 0].reshape(obins, pbins)

        return mesh_coords
        
    @profile
    def populate_2D_histograms(self, q1, q2, px, h_params, h_matrices):
        """
        Populates the density and velocity histograms simultaneously for quick runtime
        Parameters:
            beam: Beam object instance
            q1: 1st dimension's coords (must be z)
            q2: 2nd dimension's coords (must be x)
            weights: how to weight the histogram
        Returns:
            density, beta_x histograms
        """
        # Unpack h_params dict
        pbins, obins, plim, olim, filter_order, filter_window, velocity_threshold = (
            h_params["pbins"],
            h_params["obins"],
            h_params["plim"],
            h_params["olim"],
            h_params["filter_order"],
            h_params["filter_window"],
            h_params["velocity_threshold"])
        
        R, R_inv, t2 = h_matrices["R"], h_matrices["R_inv"], h_matrices["t2"]

        # Put the beam macro particle positions in the space where the grid is rotated
        particle_positions = (((np.stack((q1, q2))).T - t2) @ R_inv.T)
        #particle_positions = np.stack((q1, q2)).T

        # Create density histogram (note that we have to transfer the particle_positions to ij again)
        density = histogram_cic_2d(particle_positions[:,1], particle_positions[:,0], np.ones(len(q1)), obins, -olim*self.o_sd, olim*self.o_sd, pbins, -plim*self.p_sd, plim*self.p_sd)

        # Create a 2D DF velocity (x compononet only) distribution histogram
        beta_x = histogram_cic_2d(particle_positions[:,1], particle_positions[:,0], px, obins, -olim*self.o_sd, olim*self.o_sd, pbins, -plim*self.p_sd, plim*self.p_sd)
 
        # The minimum particle number of particles in a bin for that bin to have non zero beta_x value
        threshold = np.max(density) / velocity_threshold

        # Make each mesh element value be equal to the AVERAGE velocity of particles in said element
        # Only for bins with particle density above the threshold
        beta_x[density > threshold] /= density[density > threshold]

        # Smooth density and velocity function using 2D Savitzky-Golay filter
        density = savgol_filter(savgol_filter(x = density, window_length=filter_window, polyorder=filter_order, axis = 0), window_length=filter_window, polyorder=filter_order, axis=1)
        beta_x = savgol_filter(savgol_filter(x = beta_x, window_length=filter_window, polyorder=filter_order, axis = 0), window_length=filter_window, polyorder=filter_order, axis=1)

        # Integrate the density function over the integration space using trapezoidal rule
        o_dim = np.linspace(-olim*self.o_sd, olim*self.o_sd, obins)
        p_dim = np.linspace(-plim*self.p_sd, plim*self.p_sd, pbins)
        dsum = np.trapz(np.trapz(density, o_dim, axis=0), p_dim, axis=0)

        # Normalize the density distirbution historgram
        density /= dsum

        # Set beta_x = 0 for all bins with low particle count
        beta_x[density <= threshold] = 0

        # Create the histograms for the partial density wrt x and wrt z
        partial_density_o, partial_density_p = np.gradient(density, o_dim, p_dim)
        
        # Create the histograms for the partial beta_x wrt x
        partial_beta_o, partial_beta_p = np.gradient(beta_x, o_dim, p_dim)

        # The histograms computed via the gradient function need to be put into x and z
        partial_density = np.dstack((partial_density_p, partial_density_o))
        partial_beta = np.dstack((partial_beta_p, partial_beta_o))

        n,m,_ = partial_density.shape
        partial_density = partial_density.reshape(-1,2)
        partial_beta = partial_beta.reshape(-1,2)

        # Use transform to 
        partial_density = partial_density @ R.T
        partial_beta = partial_beta @ R.T

        partial_density = partial_density.reshape(n,m,2)
        partial_beta = partial_beta.reshape(n,m,2)

        partial_density_x = partial_density[:,:,1]
        partial_density_z = partial_density[:,:,0]
        partial_beta_x = partial_beta[:,:,1]

        return density, beta_x, partial_density_x, partial_density_z, partial_beta_x

    def plot_histogram(self, intensity_vals, title=""):
        """
        Helpful function for quickly plotting the output of the histograms onto a 2D colormap
        Parameters:
            intensity_vals: the intensity values of the histogram to be plotted (ex: self.density)
        """
        plt.figure()

        Z = self.h_coords[:, :, 1]
        X = self.h_coords[:, :, 0]

        plt.pcolormesh(Z, X, intensity_vals, shading='auto', cmap='viridis')

        plt.colorbar(label='Values') 
        plt.axis("equal")

        plt.title(title)
        plt.xlabel("z")
        plt.ylabel("x")

        plt.show()

    def plot_surface(self, intensity_vals, title=""):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        Z = self.h_coords[:, :, 1]
        X = self.h_coords[:, :, 0]

        surface = ax.plot_surface(Z, X, intensity_vals, cmap='viridis')

        # Add a color bar which maps values to colors
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

        # Labels
        ax.set_xlabel('Z axis')
        ax.set_ylabel('X axis')
        ax.set_zlabel('intensity axis')
        ax.set_title(title)
        ax.set_aspect('auto')

        plt.show()

    def index2position(self, indices, mesh_matrices):
        """
        Given mesh indices, converts to position
        Parameters:
            indices: np.array of dimension 2 (index1, index2)
        Returns:
            np.array of dimension 2 (z, x)
        """
        return (mesh_matrices["R"] @ (mesh_matrices["C"] @ (indices - mesh_matrices["t1"]))) + mesh_matrices["t2"]
    
    def position2index(self, point, mesh_matrices):
        """
        Given an z and x position, gives the indices of the four closest mesh points
        Parameters:
            position: np.array of dimension 2 (z,x)
        Returns:
            np.array(upper_left, lower_left, upper_right, lower_right)
        """
        # Transform the point into coordinates space
        coord_position = (mesh_matrices["C_inv"] @ (mesh_matrices["R_inv"] @ (point - mesh_matrices["t2"]))) + mesh_matrices["t1"]

        return np.array([[math.floor(coord_position[0]), math.ceil(coord_position[1])],
                         [math.floor(coord_position[0]), math.floor(coord_position[1])],
                         [math.ceil(coord_position[0]), math.ceil(coord_position[1])],
                         [math.ceil(coord_position[0]), math.floor(coord_position[1])]])

    def plot_grid_transformation(self, beam, mesh_params, mesh_coords, fig=None, ax=None, title=""):
        """
        Plots the mesh ontop of the beam distribution
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        X = mesh_coords[:, :, 0]
        Z = mesh_coords[:, :, 1]
        
        #Plot the mesh grid
        self.plot_mesh(ax, Z, X, "blue")

        # Plot the beam
        ax.scatter(beam.z, beam.x, color="red", s=10, label="Beam Distribution")

        # Plot the line of best fit
        z_end = mesh_params["plim"]*self.p_sd * np.cos(self.tilt_angle)
        x_end = mesh_params["plim"]*self.p_sd * np.sin(self.tilt_angle)
        #ax.plot([-z_end, z_end], [-x_end, x_end], color="black", linewidth = 2, label = "line of best fit")

        # Plot the line orthonormal to the line of best fit
        z_end = mesh_params["olim"]*self.o_sd * np.cos(self.tilt_angle + np.pi/2)
        x_end = mesh_params["olim"]*self.o_sd * np.sin(self.tilt_angle + np.pi/2)
        #ax.plot([-z_end, z_end], [-x_end, x_end], color="purple", linewidth = 2, label = "line ortho to best fit")

        if title == "":
            title = "Scatter Plot of Beam Distribution with Mesh Overlay"

        ax.set_title(title)
        ax.set_xlabel("z")
        ax.set_ylabel("x")
        ax.axis("equal")
        ax.legend()
        
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