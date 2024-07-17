# Import standard library modules

# Import third-party modules
import numpy as np
import matplotlib.pyplot as plt

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
        for step_index, snapshot in enumerate(self.step_snapshots):
            # Propagate beam -> Beam class
            # Populate snapshot -> Step_Snapshot class
            # Get CSR mesh
            # Create integration mesh
            # Compute integrand
            # Compute wake via trapezoidal integration
            # Apply to beam
            #print(step_index)
            pass
        
        #"""

        #""""
        self.step_snapshots[0].populate(self.beam)
        self.step_snapshots[0].plot_histogram(self.step_snapshots[0].density, "density")
        self.step_snapshots[0].plot_surface(self.step_snapshots[0].density, "density")
        self.step_snapshots[0].plot_histogram(self.step_snapshots[0].beta_x, "beta_x")
        self.step_snapshots[0].plot_surface(self.step_snapshots[0].beta_x, "beta_x")
        self.step_snapshots[0].plot_histogram(self.step_snapshots[0].partial_density_x, "partial_density_x")
        self.step_snapshots[0].plot_surface(self.step_snapshots[0].partial_density_x, "partial_density_x")
        self.step_snapshots[0].plot_histogram(self.step_snapshots[0].partial_density_z, "partial_density_z")
        self.step_snapshots[0].plot_surface(self.step_snapshots[0].partial_density_z, "partial_density_z")
        self.step_snapshots[0].plot_histogram(self.step_snapshots[0].partial_beta_x, "partial_beta_x")
        self.step_snapshots[0].plot_surface(self.step_snapshots[0].partial_beta_x, "partial_beta_x")
        

        print(np.tan(self.step_snapshots[0].tilt_angle))
        #"""


        """
        self.step_snapshots[0].populate(self.beam)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        data_Jingyi = np.load("data_program1.npz")
        Z1 = data_Jingyi["Z"]
        X1 = data_Jingyi["X"]
        density1 = data_Jingyi["intensity_vals"]

        Z2 = self.step_snapshots[0].mesh_coords[:, :, 1]
        X2 = self.step_snapshots[0].mesh_coords[:, :, 0]
        density2 = self.step_snapshots[0].density

        # Plot the first dataset
        surface1 = ax.plot_surface(Z1, X1, density1, cmap='viridis', alpha=0.5)

        # Plot the second dataset
        surface2 = ax.plot_surface(Z2, X2, density2, cmap='plasma', alpha=0.5)

        # Labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')

        plt.show()
        """

        #self.step_snapshots[0].plot_grid_transformation(self.beam)

        

        """
        self.step_snapshots[0].populate(self.beam)
        self.step_snapshots[0].plot_grid_transformation(self.beam)
        print(np.tan(self.step_snapshots[0].tilt_angle))

        """
        for a in self.lattice.bmadx_elements:
            print(a)
            print("")



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
 


