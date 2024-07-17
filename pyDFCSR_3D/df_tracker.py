import numpy as np

from lattice import Lattice
from beam import Beam
from step_snapshot import Step_Snapshot

class DF_Tracker:
    """
    The class for the 'distribution function' (DF) of a beam
    Doesn't handle the discrete data of the DF, rather it stores and compute the beam characteristics (ie tracker)
    """
    def __init__(self, lattice, input_dic={}):
        """
        Initializes the DF tracker
        Parameters:
            initial_beam: Beam object of the initial beam
            lattice: Lattice object
            input_dic: optional, a dictionary mapping various optional parameters (keys) to
                       their values that can be specified if desired
        Returns:
            Instance of DF_Tracker
        """
        # Initialize the optional parameters
        self.configure_params(**input_dic)

        # Initialize all Step_Snapshot objects
        self.initialize_snapshots(lattice)

    def configure_params(self, pbins=100, obins=100, plim=5, olim=5,
                         filter_order=0, filter_window=0,
                         velocity_threhold=5):
        """
        Initalizes the optional parameters
        """

        # Number of bins allocated for the histograms in the direction parallel and orthonormal to the beam
        self.pbins = pbins
        self.obins = obins

        # The number of standard deviations away from the mean (in both directions) for the histograms
        self.plim = plim
        self.olim = olim

        # 1/(proportion of the highest occupied bin)
        # TODO: make this a proportion
        self.velocity_threhold = velocity_threhold

        # Savitzky-Golay filter settings
        self.polyorder = filter_order # The order of the polynomial used to fit the samples
        self.window_length = filter_window # The length of the filter_window, must be positive and odd

    def initialize_snapshots(self, lattice):
        """
        Initializes all snapshot instances which preallocates all np arrays
        """
        self.snapshots = [None]*lattice.total_steps

        # For each step initialize a snapshot
        for step_index in range(len(self.snapshots)):
            self.snapshots[step_index] = Step_Snapshot(self.pbins, self.obins, self.plim, self.olim)

    def populate_first_snapshot(self, beam):
        """
        Test function
        """
        self.snapshots[0].populate(beam, self.window_length, self.polyorder, self.velocity_threhold)
        #self.snapshots[0].plot_grid_transformation(beam)
        self.snapshots[0].plot_histogram(self.snapshots[0].density, "density")
        self.snapshots[0].plot_histogram(self.snapshots[0].beta_x, "beta_x")
        self.snapshots[0].plot_histogram(self.snapshots[0].partial_density_x, "partial_density_x")
        self.snapshots[0].plot_histogram(self.snapshots[0].partial_density_z, "partial_density_z")
        self.snapshots[0].plot_histogram(self.snapshots[0].partial_beta_x, "partial_beta_x")






