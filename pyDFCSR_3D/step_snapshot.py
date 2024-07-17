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

    def __init__(self, pbins=100, obins=100, plim=5, olim=5, filter_order=1, filter_window=5, velocity_threshold=1000):
        # Mesh dimensions
        self.pbins = pbins
        self.obins = obins
        self.plim = plim
        self.olim = olim

        # Histogram settings
        self.polyorder = filter_order
        self.window_length = filter_window
        self.velocity_threshold = velocity_threshold

        # Positions of each vertex of the mesh
        self.mesh_coords = np.empty((obins, pbins, 2), dtype=np.float64)

        # Characteristics of beam distribution at this snapshot
        self.x_mean = None
        self.z_mean = None
        self.p_sd = None
        self.o_sd = None
        self.tilt_angle = None  # angle of the line of best fit from the horizontal axis

        # The s value of the step on the reference trajectory
        self.s_val = None

        # Initialize matrices describing the mesh transformation from the unit mesh
        self.t1 = np.empty(2, dtype=np.float64)             # 1st tranform (in unit mesh)
        self.C = np.empty((2,2), dtype=np.float64)          # Compression
        self.C_inv = np.empty((2,2), dtype=np.float64)      # Compression inversion
        self.R = np.empty((2,2), dtype=np.float64)          # Rotation
        self.R_inv = np.empty((2,2), dtype=np.float64)      # Rotation inversion
        self.t2 = np.empty(2, dtype=np.float64)             # 2nd tranform (to beam center)

        # Preallocate space for the numpy arrays (histograms)
        self.density = np.empty((obins, pbins), dtype=np.float64)            # Density
        self.partial_density_x = np.empty((obins, pbins), dtype=np.float64)  # Partial Derivative of density wrt x
        self.partial_density_z = np.empty((obins, pbins), dtype=np.float64)  # Partial Derivative of density wrt x
        self.beta_x = np.empty((obins, pbins), dtype=np.float64)             # x component of beta = v/c
        self.partial_beta_x = np.empty((obins, pbins), dtype=np.float64)     # x component of partial beta

    def populate(self, beam):
        """
        Converts a Beam object into a fully populated Step_Snapshot object
        Parameters:
            beam: Beam object to be used
            window_length & polyorder: Savitzky-Golay filter parameters
        """
        # Populate beam characteristics
        self.x_mean = beam.mean_x
        self.z_mean = beam.mean_z
        self.p_sd, self.o_sd = beam.get_std_wrt_linear_fit()
        self.tilt_angle = beam.tilt_angle

        # Popultate transformation matrices (not in ij format)
        self.t1 = np.array([(self.pbins-1)/2, (self.obins-1)/2], dtype = np.float64) 
        self.C = np.array([[2*(self.plim * self.p_sd)/(self.pbins-1), 0.0],
                          [0.0, 2*(self.olim * self.o_sd)/(self.obins-1)]], dtype = np.float64)
        self.C_inv = np.linalg.inv(self.C)
        self.R = np.array([[np.cos(self.tilt_angle), -np.sin(self.tilt_angle)],
                          [np.sin(self.tilt_angle), np.cos(self.tilt_angle)]], dtype=np.float64)
        self.R_inv = np.linalg.inv(self.R)
        self.t2 = np.array([self.z_mean,self.x_mean], dtype=np.float64)

        # Populate mesh coordinates
        self.mesh_coords = self.populate_mesh_coordinates()

        # Populate density and beta_x histograms
        (self.density, 
         self.beta_x,
         self.partial_density_x,
         self.partial_density_z,
         self.partial_beta_x) = self.populate_2D_histograms(beam.z, beam.x, beam.px, self.window_length, self.polyorder, self.velocity_threshold)

    def populate_mesh_coordinates(self):
        """
        Helper function to populate, populates the mesh coordinates
        Parameters:
            mesh_coordinates: the mesh_coordinates attribute to be populated
        """
        # Seed the mesh
        mesh_z = np.arange(self.pbins)
        mesh_x = np.arange(self.obins)

        Z, X = np.meshgrid(mesh_z, mesh_x)
        mesh_coords_list = np.stack((Z.flatten(), X.flatten()), axis=0).T

        # Shift mesh to center
        mesh_coords_list = mesh_coords_list - self.t1
        
        # Compress mesh
        mesh_coords_list = mesh_coords_list @ self.C.T

        # Rotate mesh
        mesh_coords_list = mesh_coords_list @ self.R.T

        # Translate mesh to center of beam distribution
        mesh_coords_list = mesh_coords_list + self.t2

        # Populate the mesh coordinates in ij format
        mesh_coords = np.empty((self.obins, self.pbins, 2), dtype=np.float64)
        mesh_coords[:,:,0] = mesh_coords_list[:, 1].reshape(self.obins, self.pbins)
        mesh_coords[:,:,1] = mesh_coords_list[:, 0].reshape(self.obins, self.pbins)

        return mesh_coords
        
    @profile
    def populate_2D_histograms(self, q1, q2, px, window_length, polyorder, velocity_threshold):
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
        # Put the beam macro particle positions in the space where the grid is rotated
        particle_positions = (((np.stack((q1, q2))).T - self.t2) @ self.R_inv.T)

        # Create density histogram (note that we have to transfer the particle_positions to ij again)
        density = histogram_cic_2d(particle_positions[:,1], particle_positions[:,0], np.ones(len(q1)), self.obins, -self.olim*self.o_sd, self.olim*self.o_sd, self.pbins, -self.plim*self.p_sd, self.plim*self.p_sd)

        # Create a 2D DF velocity (x compononet only) distribution histogram
        beta_x = histogram_cic_2d(particle_positions[:,1], particle_positions[:,0], px, self.obins, -self.olim*self.o_sd, self.olim*self.o_sd, self.pbins, -self.plim*self.p_sd, self.plim*self.p_sd)

        # Smooth density and velocity function using 2D Savitzky-Golay filter
        density = savgol_filter(savgol_filter(x = density, window_length=window_length, polyorder=polyorder, axis = 0), window_length=window_length, polyorder=polyorder, axis=1)
        beta_x = savgol_filter(savgol_filter(x = beta_x, window_length=window_length, polyorder=polyorder, axis = 0), window_length=window_length, polyorder=polyorder, axis=1)

        # Integrate the density function over the integration space using trapezoidal rule
        o_dim = np.linspace(-self.olim*self.o_sd, self.olim*self.o_sd, self.obins)
        p_dim = np.linspace(-self.plim*self.p_sd, self.plim*self.p_sd, self.pbins)
        dsum = np.trapz(np.trapz(density, o_dim, axis=0), p_dim, axis=0)

        # Normalize the density distirbution historgram
        density /= dsum

        # The minimum particle number of particles in a bin for that bin to have non zero beta_x value
        threshold = np.max(density) / velocity_threshold

        # Set beta_x = 0 for all bins with low particle count
        beta_x[density <= threshold] = 0

        # Create the histograms for the partial density wrt x and wrt z
        partial_density_o, partial_density_p = np.gradient(density, o_dim, p_dim)
        
        # Create the histograms for the partial beta_x wrt x
        partial_beta_o, partial_beta_p = np.gradient(beta_x, o_dim, p_dim)

        # The histograms compute via the gradient function need to be put into x and z
        partial_density = np.dstack((partial_density_p, partial_density_o))
        partial_beta = np.dstack((partial_beta_p, partial_beta_o))

        n,m,_ = partial_density.shape
        partial_density = partial_density.reshape(-1,2)
        partial_beta = partial_beta.reshape(-1,2)

        partial_density = partial_density @ self.R.T
        partial_beta = partial_beta @ self.R.T

        partial_density = partial_density.reshape(n,m,2)
        partial_beta = partial_beta.reshape(n,m,2)

        partial_density_x = partial_density[:,:,1]
        partial_density_z = partial_density[:,:,0]
        partial_beta_x = partial_beta[:,:,1]

        return density, beta_x, partial_density_x, partial_density_z, partial_beta_x

    def position2index(self, point):
        """
        Given an z and x position, gives the indices of the four closest mesh points
        Parameters:
            position: np.array of dimension 2 (z,x)
        Returns:
            np.array(upper_left, lower_left, upper_right, lower_right)
        """
        # Transform the point into coordinates space
        coord_position = (self.C_inv @ (self.R_inv @ (point - self.t2))) + self.t1

        return np.array([[math.floor(coord_position[0]), math.ceil(coord_position[1])],
                         [math.floor(coord_position[0]), math.floor(coord_position[1])],
                         [math.ceil(coord_position[0]), math.ceil(coord_position[1])],
                         [math.ceil(coord_position[0]), math.floor(coord_position[1])]])

    def plot_histogram(self, intensity_vals, title=""):
        """
        Helpful function for quickly plotting the output of the histograms onto a 2D colormap
        Parameters:
            intensity_vals: the intensity values of the histogram to be plotted (ex: self.density)
        """
        plt.figure()

        Z = self.mesh_coords[:, :, 1]
        X = self.mesh_coords[:, :, 0]

        #X = np.linspace(-self.olim*self.o_sd, self.olim*self.o_sd, self.obins)
        #Z = np.linspace(-self.plim*self.p_sd, self.plim*self.p_sd, self.pbins)

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

        Z = self.mesh_coords[:, :, 1]
        X = self.mesh_coords[:, :, 0]

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

    def index2position(self, indices):
        """
        Given mesh indices, converts to position
        Parameters:
            indices: np.array of dimension 2 (index1, index2)
        Returns:
            np.array of dimension 2 (z, x)
        """
        return (self.R @ (self.C @ (indices - self.t1))) + self.t2

    def plot_grid_transformation(self, beam, fig=None, ax=None):
        """
        Plots the mesh ontop of the beam distribution
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        X = self.mesh_coords[:, :, 0]
        Z = self.mesh_coords[:, :, 1]
        
        #Plot the mesh grid
        #self.plot_mesh(ax, Z, X, "blue")

        # Plot the beam
        ax.scatter(beam.z, beam.x, color="red", s=10, label="Beam Distribution")

        # Plot the line of best fit
        z_end = self.plim*self.p_sd * np.cos(self.tilt_angle)
        x_end = self.plim*self.p_sd * np.sin(self.tilt_angle)
        #ax.plot([-z_end, z_end], [-x_end, x_end], color="black", linewidth = 2, label = "line of best fit")

        # Plot the line orthonormal to the line of best fit
        z_end = self.olim*self.o_sd * np.cos(self.tilt_angle + np.pi/2)
        x_end = self.olim*self.o_sd * np.sin(self.tilt_angle + np.pi/2)
        #ax.plot([-z_end, z_end], [-x_end, x_end], color="purple", linewidth = 2, label = "line ortho to best fit")

        ax.set_title("Scatter Plot of Beam Distribution with Mesh Overlay")
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