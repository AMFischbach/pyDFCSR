import sys

import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

sys.path.insert(0, '/Users/amf16/pyDFCSR/fast_pyDFCSR_2D')
from beam import Beam

"""
Module Name: post_processor.py

Helps with data processing of already computed CSR wake data
"""

def load_new_data(filepath, load_histograms=False):
    """
    Given a filepath to an h5 file, unpacks it
    Parameters:
        filepath: string, the filepath of the data file
    Returns:
        step_snapshots and statistics
    """
    if load_histograms:
        step_snapshots = []
    statistics = {}

    with h5py.File(filepath, 'r') as f:
            
        # Load statistics
        stats_grp = f['statistics']
        for key in stats_grp.keys():
            item = stats_grp[key]
            if isinstance(item, h5py.Group):
                statistics[key] = {}
                for sub_key in item.keys():
                    statistics[key][sub_key] = item[sub_key][:]
            else:
                statistics[key] = item[:]

    return statistics

def load_old_data(filepath):
    statistics = {}

    with h5py.File(filepath, 'r') as hf:
        # Read top-level datasets
        step_positions = hf['step_positions'][:]
        coords = hf['coords'][:]
        n_vec = hf['n_vec'][:]
        tau_vec = hf['tau_vec'][:]
        
        # Read nested dictionary
        statistics = hdf5_to_dict(hf)

    return statistics

def hdf5_to_dict(hf, group=None):
    if group is None:
        group = hf

    dic = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            dic[key] = hdf5_to_dict(hf, item)
        else:
            dic[key] = item[:]
    return dic

def file2histograms(pbins, obins, plim, olim, file_path, filter_order=2, filter_window=5, velocity_threshold=1000):
    """
    Create the density, beta_x, etc histograms on a mesh
    Parameters:
        pbins, obins: number of bins in the parallel and orthonormal directions
        plim, olim: number of std away from the beam center to extend the mesh
        file_path: the file path to the Particle_Group File
    Returns:
        dict of histograms which also includes mesh_coords
    """
    # Initalize inputs
    input_beam = {}
    input_beam["style"] = "ParticleGroup"
    input_beam["ParticleGroup_h5"] = file_path

    # The histogram mesh format
    histo_mesh_params = {}
    histo_mesh_params["pbins"] = pbins
    histo_mesh_params["obins"] = obins
    histo_mesh_params["plim"] = plim
    histo_mesh_params["olim"] = olim
    histo_mesh_params["filter_order"] = filter_order
    histo_mesh_params["filter_window"] = filter_window
    histo_mesh_params["velocity_threshold"] = velocity_threshold

    # We populate this but we NEVER use it
    CSR_mesh_params = {}
    CSR_mesh_params["pbins"] = pbins
    CSR_mesh_params["obins"] = obins
    CSR_mesh_params["plim"] = plim
    CSR_mesh_params["olim"] = olim

    # Create a Beam object
    beam = Beam(input_beam)

    # Create a Step_Snapshot object and populate it with the beam
    step_snapshot = Step_Snapshot(0, histo_mesh_params, CSR_mesh_params)
    step_snapshot.populate(beam, update_CSR_mesh = False)

    # Create the histogram dict to be returned
    histograms = {}
    histograms["density"] = step_snapshot.density
    histograms["beta_x"] = step_snapshot.beta_x
    histograms["partial_density_x"] = step_snapshot.partial_density_x
    histograms["partial_density_z"] = step_snapshot.partial_density_z
    histograms["partial_beta_x"] = step_snapshot.partial_beta_x
    histograms["mesh_coords"] = step_snapshot.h_coords

    return histograms

def plot_surface(step_snapshot, surface_type, fig=None, ax=None, alpha=1, color="blue", title=""):
    """
    Given a step_snapshot loaded from the h5 file, is able to plot the surface on the mesh
    """
    show = False
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        show = True

    Z = step_snapshot["h_coords"][:, :, 1]
    X = step_snapshot["h_coords"][:, :, 0]

    # Remove all non_zero intensity vals
    I_masked = np.ma.masked_equal(step_snapshot[surface_type], 0)

    ax.plot_surface(Z, X, I_masked, color=color, alpha=alpha)

    # Labels
    ax.set_xlabel('Z axis')
    ax.set_ylabel('X axis')
    ax.set_zlabel('intensity axis')
    ax.set_title(title)
    ax.set_aspect('auto')

    if show:
        plt.show()

def plot_surface2(mesh_coords, histogram_vals, fig=None, ax=None, alpha=1, color="blue", label=""):
    """
    Given a step_snapshot loaded from the h5 file, is able to plot the surface on the mesh
    """
    show = False
    
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        show = True

    Z = mesh_coords[:, :, 1]
    X = mesh_coords[:, :, 0]

    # Remove all non_zero intensity vals
    I_masked = np.ma.masked_equal(histogram_vals, 0)

    ax.plot_surface(Z, X, I_masked, color=color, alpha=alpha, label=label)

    if show:
        ax.set_xlabel('Z axis')
        ax.set_ylabel('X axis')
        ax.set_zlabel('intensity axis')
        ax.set_aspect('auto')
        ax.legend()
        plt.show()

def plot_histogram(step_snapshot, histogram_type, title=""):
    plt.figure()

    Z = step_snapshot["h_coords"][:, :, 1]
    X = step_snapshot["h_coords"][:, :, 0]

    plt.pcolormesh(Z, X, step_snapshot[histogram_type], shading='auto', cmap='viridis')

    plt.colorbar(label='Values') 
    plt.axis("equal")

    plt.title(histogram_type)
    plt.xlabel("z")
    plt.ylabel("x")
    plt.title(title)

    plt.show()

def plot_histogram2(mesh_coords, histogram_vals, title="", ax=None, fig=None):
    Z = mesh_coords[:, :, 1]
    X = mesh_coords[:, :, 0]

    if ax is None or fig is None:
        plt.figure()

        plt.pcolormesh(Z, X, histogram_vals, shading='auto', cmap='viridis')

        plt.colorbar(label='Values') 
        plt.axis("equal")

        plt.xlabel("z")
        plt.ylabel("x")
        plt.title(title)
        plt.show()
    
    else:
        c = ax.pcolormesh(Z, X, histogram_vals, shading="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        fig.colorbar(c, ax=ax, orientation="vertical")

def plot_statistic(statistic, title="", new_figure=True, label="", color="red"):
    if new_figure:
        plt.figure()

    x_vals = np.arange(1, len(statistic)+1)
    plt.plot(x_vals, statistic, label=label, color=color, linestyle="-", marker="o")

    if new_figure:
        plt.title(title)
        plt.xlabel("step number")
        plt.show()

def create_beam_gif(step_snapshots):
    """
    Creates an gif showing how the beam propagates with time
    """
    fig, ax = plt.subplots()
    Z = step_snapshots[0]["h_coords"][:, :, 1]
    X = step_snapshots[0]["h_coords"][:, :, 0]
    
    pcm = ax.pcolormesh(Z, X, step_snapshots[0]["density"], shading='auto', cmap='viridis')
    cb = plt.colorbar(pcm, ax=ax, label='Values')
    #plt.axis("equal")
    plt.xlabel("z")
    plt.ylabel("x")

    animate = lambda frame: update(frame, cb, ax, step_snapshots)

    ani = animation.FuncAnimation(fig, animate, frames=len(step_snapshots), interval=200, blit=True)

    # Save the animation as a GIF file
    ani.save('pcolormesh_animation3.gif', writer='imagemagick', fps=2)

    """
    mesh_coords_list = [None]*len(step_snapshots)
    density_list = [None]*len(step_snapshots)
    for index, step_snapshot in enumerate(step_snapshots):
        mesh_coords_list[index] = step_snapshot.mesh_coords
        density_list[index] = step_snapshot.density
    """
    
# Function to update the plot for each frame
def update(frame, cb, ax, step_snapshots):
    ax.clear()  # Clear the previous plot

    mesh_coords = step_snapshots[frame]["h_coords"]
    intensity_vals = step_snapshots[frame]["density"]

    pcm = ax.pcolormesh(mesh_coords[:, :, 1], mesh_coords[:, :, 0], intensity_vals, shading='auto', cmap='viridis')
    
    cb.update_normal(pcm)
    #plt.axis("equal")
    plt.title("beam distribution at step "+str(frame))
    plt.xlabel("z")
    plt.ylabel("x")

    return pcm,

