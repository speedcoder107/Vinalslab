import numpy as np
cimport numpy as np
import isinggame as ig
import random
from scipy.constants import Boltzmann
from scipy.ndimage import convolve, generate_binary_structure
from cython cimport wraparound
import matplotlib.pyplot as plt
from ipywidgets import interact
from matplotlib.animation import FuncAnimation
import os
import imageio


# constants
k_b = Boltzmann

cdef class full_model:
    
    cdef public np.ndarray lattice
    cdef public np.ndarray B_field
    cdef public double beta
    cdef public double time
    cdef public double net_energy  # Change the type to double
    cdef public double net_spin

    def __init__(self, np.ndarray[np.int32_t, ndim=2] lattice, np.ndarray[np.int32_t, ndim=2] B_field, double beta, np.ndarray[np.int32_t, ndim=1] net_energy=None, np.ndarray[np.int32_t, ndim=1] net_spin=None):
        '''
        Initializes an instance of the Ising model, providing a full picture of its configuration.

        Parameters: 
        - lattice (numpy.ndarray): Initial configuration of the lattice.
        - B_field (float): External magnetic field strength applied to the lattice.
        - temp (float): Temperature of the lattice system.

        Returns:
        - None
        '''
        self.lattice = lattice  # Initialize the lattice configuration
        self.B_field = B_field  # Initialize the external magnetic field strength
        self.beta = beta        # Initialize the temperature of the lattice system
        self.net_energy = 0.0   # Initialize net_energy to 0.0
        self.net_spin = 0.0


def field_generator(int row, int col, *args):
    
    '''
    Generates a field with 'row' number of rows and 'col' number of columns. The optional value, fixes the values of the field

    Parameters:
    - row (int): number of rows of the field.
    - col (int): number of columns of the field.
    - *args (int): fixes the values of all elements of the field to the given value
    
    Returns:
    - 2D array of the field.
    '''

    cdef np.ndarray[np.int_t, ndim=2] field
    if not args:
        field = np.random.choice([1, -1], size=(row, col))
    else:
        field = np.random.choice(args, size=(row, col))
    return field.astype(np.int32)  # Convert to the desired data type


cpdef bint set_net_spin(full_model model):
    
    """
    Calculates the net spin of the lattice and stores it inside the full_model class of the given parameter.

    Parameters:
    - model (full_model): The entire ising model with all the information

    Returns:
    - None

    Raises:
    Exception: if it is unable to set energy
    """

    cdef np.ndarray[np.int_t, ndim=2] lattice = model.lattice
    try:
        model.net_spin = np.sum(lattice)
        return True
    except:
        return False


cpdef double get_net_energy(full_model model):
    """
    Retrieves the net energy of the Ising model.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.

    Returns:
    - float: The net energy of the Ising model.
    """
    return model.net_energy

cpdef double get_net_spin(full_model model):
    """
    Retrieves the net spin of the lattice in the Ising model.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.

    Returns:
    - float: The net spin of the lattice in the Ising model.
    """
    return model.net_spin

cpdef np.ndarray[np.int_t, ndim=2] get_lattice(full_model model):
    """
    Retrieves the lattice of spins in the Ising model.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.

    Returns:
    - array-like: The lattice of spins in the Ising model.
    """
    return model.lattice

cpdef double get_B_field(full_model model):
    """
    Retrieves the external magnetic field strength applied to the Ising model.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.

    Returns:
    - float: The strength of the external magnetic field applied to the Ising model.
    """
    return model.B_field

cpdef double get_temp(full_model model):
    """
    Retrieves the temperature of the Ising model.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.

    Returns:
    - float: The temperature of the Ising model.
    """

    return 1 / (model.beta * k_b)

cpdef np.ndarray[np.int_t, ndim=2] get_neighborhood(np.ndarray[np.int_t, ndim=2] lattice, int row, int col):
    '''
    Extracts the neighborhood of an element in a 2D NumPy array.

    Parameters:
    - lattice (2D array): Input 2D NumPy array representing the lattice.
    - n (int): Row index of the element whose neighborhood is to be extracted.
    - m (int): Column index of the element whose neighborhood is to be extracted.

    Returns:
    - 2D array: Array containing the neighborhood elements surrounding the specified element.
    '''

    # Get the dimensions of the lattice
    cdef int rows = lattice.shape[0]
    cdef int cols = lattice.shape[1]
    
    # Define the indices of the neighborhood elements
    cdef int row_indices[3]
    row_indices[0] = row - 1
    row_indices[1] = row
    row_indices[2] = row + 1
    
    cdef int col_indices[3]
    col_indices[0] = col - 1
    col_indices[1] = col
    col_indices[2] = col + 1
    
    # Initialize the neighborhood array
    cdef np.ndarray[np.int_t, ndim=2] neighborhood = np.empty((3, 3), dtype=lattice.dtype)
    
    # Iterate over the indices and extract neighborhood elements
    cdef int i, j
    for i in range(3):
        for j in range(3):
            # Ensure the indices are within the bounds of the lattice
            if 0 <= row_indices[i] < rows and 0 <= col_indices[j] < cols:
                # If indices are within bounds, extract the corresponding element
                neighborhood[i, j] = lattice[row_indices[i], col_indices[j]]
            else:
                # If indices are out of bounds, use a placeholder value (0)
                neighborhood[i, j] = 0
    
    return neighborhood


def set_net_energy(model):
    
    '''
    Calculates the energy of the lattice using a simple Ising model.

    Parameters:
    - lattice (numpy.ndarray): 2D array representing the lattice configuration.

    Returns:
    - energy (float): Total energy of the lattice configuration.

    Source/inspiration:
    - https://github.com/lukepolson/youtube_channel/blob/main/Python%20Metaphysics%20Series/vid14.ipynb
    '''
    try:
        kern = generate_binary_structure(2, 1) 
        kern[1][1] = False
        arr = -model.lattice * convolve(model.lattice, kern, mode='constant', cval=0) - model.lattice * model.B_field
        model.net_energy = arr.sum()
        return True
    except Exception:
        return False


cpdef double get_energy(full_model model, int row, int col):
    """
    Calculates the energy of a spin at the specified row and column in the lattice of the Ising model.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.
    - row (int): The row index of the spin in the lattice.
    - col (int): The column index of the spin in the lattice.

    Returns:
    - float: The energy of the spin at the specified position.

    Source/inspiration:
    - https://github.com/lukepolson/youtube_channel/blob/main/Python%20Metaphysics%20Series/vid14.ipynb
    """
    # Declare typed variables using Cython
    cdef np.ndarray[np.int_t, ndim=2] lattice = model.lattice  # Declare lattice as a 2D array of integers
    cdef np.ndarray[np.int_t, ndim=2] l_neighbor = get_neighborhood(lattice, row, col)  # Get the neighborhood of the spin
    cdef double elem = lattice[row, col]  # Get the value of the spin at the specified position
    cdef np.ndarray[np.int_t, ndim=2] kern = generate_binary_structure(2, 1).astype(np.int32)  # Generate a binary kernel
    kern[1, 1] = 0  # Set the central element of the kernel to False

    cdef double energy  # Declare energy variable
    
    # Calculate energy using the Ising model formula
    energy = -elem * convolve(l_neighbor, kern, mode='constant', cval=0)[1,1] - elem * model.B_field[row, col]
    return energy


# Cythonize the functions
cpdef int get_spin(full_model model, int row, int col):
    """
    Retrieves the spin value at the specified row and column in the lattice of the Ising model.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.
    - row (int): The row index of the spin in the lattice.
    - col (int): The column index of the spin in the lattice.

    Returns:
    - float: The spin value at the specified position.
    """

    return model.lattice[row, col]

cpdef bint metropolis(full_model model):
    """
    Performs a single Metropolis step in the Ising model simulation.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.

    Returns:
    - bint: True if the spin was flipped, False otherwise.
    """
    # Declare typed variables using Cython
    cdef np.ndarray[np.int_t, ndim=2] lattice = model.lattice  # Declare lattice as a 2D array of integers
    cdef int row = random.randint(0, lattice.shape[0]-1)  # Randomly select a row index
    cdef int col = random.randint(0, lattice.shape[1]-1)  # Randomly select a column index
    cdef int spin_i = get_spin(model, row, col)  # Get the initial spin value at the selected position
    cdef int spin_f = -spin_i  # Flip the spin to its opposite value
    cdef double energy_i = get_energy(model, row, col)  # Calculate the initial energy at the selected position
    cdef double energy_f = -energy_i  # Calculate the final energy after flipping the spin
    cdef double d_energy = energy_f - energy_i  # Calculate the change in energy
    cdef int d_spin = spin_f - spin_i  # Calculate the change in spin
    
    # Perform the Metropolis acceptance criterion
    if d_energy < 0:
        # If the energy decreases, always accept the spin flip
        lattice[row, col] = spin_f
        model.net_energy += d_energy
        model.net_spin += d_spin   
        return True
    elif np.random.random() < np.exp(-model.beta * d_energy):
        # If the energy increases, accept the spin flip with a probability based on the Boltzmann factor
        lattice[row, col] = spin_f
        model.net_energy += d_energy
        model.net_spin += d_spin   
        return True
    else: 
        # If the spin flip is not accepted, return False
        return False

cpdef bint my_algorithm(full_model model, int row, int col):
    """
    Performs a single Metropolis step in the Ising model simulation.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.

    Returns:
    - bint: True if the spin was flipped, False otherwise.
    """
    # Declare typed variables using Cython
    cdef np.ndarray[np.int_t, ndim=2] lattice = model.lattice  # Declare lattice as a 2D array of integers
    cdef int spin_i = get_spin(model, row, col)  # Get the initial spin value at the selected position
    cdef int spin_f = -spin_i  # Flip the spin to its opposite value
    cdef double energy_i = get_energy(model, row, col)  # Calculate the initial energy at the selected position
    cdef double energy_f = -energy_i  # Calculate the final energy after flipping the spin
    cdef double d_energy = energy_f - energy_i  # Calculate the change in energy
    cdef int d_spin = spin_f - spin_i  # Calculate the change in spin
    
    # Perform the Metropolis acceptance criterion
    if d_energy < 0:
        # If the energy decreases, always accept the spin flip
        lattice[row, col] = spin_f
        model.net_energy += d_energy
        model.net_spin += d_spin   
        return True
    elif np.random.random() < np.exp(-model.beta * d_energy):
        # If the energy increases, accept the spin flip with a probability based on the Boltzmann factor
        lattice[row, col] = spin_f
        model.net_energy += d_energy
        model.net_spin += d_spin   
        return True
    else: 
        # If the spin flip is not accepted, return False
        return False



def display_spin_field(field):
    """
    Converts a spin field to a grayscale image for display.

    Parameters:
    - field (numpy.ndarray): The spin field to be converted.

    Returns:
    - numpy.ndarray: Grayscale image representing the spin field.
    """

    image = np.uint8((field + 1) * 0.5 * 255)
    return image

def display_ising_sequence(images):
    """
    Displays an interactive sequence of images representing the evolution of an Ising model simulation.

    Parameters:
    - images (list of numpy.ndarray): List of spin field images representing different simulation steps.

    Returns:
    - None
    """    
    def _show(frame=(0, len(images) - 1)):
        plt.figure(figsize=(6, 6))
        plt.imshow(display_spin_field(images[frame]), cmap='gray')
        plt.axis('off')
        plt.show()
    return interact(_show)

def graph_list(data, x_label, y_label, title):
    """
    Graphs a list of numbers using Matplotlib.

    Parameters:
        data (list): The input list containing numbers.
    """
    # Generate x-values (assuming data is evenly spaced)
    x_values = list(range(1, len(data) + 1))
    
    # Plot the data
    plt.plot(x_values, data, marker='o', linestyle='-')
    
    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # Show the plot
    plt.grid(True)
    plt.show()

def convert_to_extension(filename, extension):
    """
    Convert a file to have a specified extension.

    Parameters:
    - filename (str): The name of the file.
    - extension (str): The desired extension.

    Returns:
    - str: The new filename with the desired extension.

    Raises:
    - None
    """
    # Split the filename and its extension
    name, ext = os.path.splitext(filename)

    # Check if the extension is already .html
    if ext == '.' + extension:
        return filename  # Nothing to do, return the original filename

    # Replace the current extension with .html
    new_filename = name + '.' + extension

    # Rename the file if it exists
    if os.path.exists(filename):
        os.rename(filename, new_filename)
        print(f"File '{filename}' renamed to '{new_filename}'")
    return new_filename

def display_spin_field_html(field):
    """
    Converts a spin field to a grayscale image for display.

    Parameters:
    - field (numpy.ndarray): The spin field to be converted.

    Returns:
    - numpy.ndarray: Grayscale image representing the spin field.
    """    
    image = np.uint8((field + 1) * 0.5 * 255)
    return image

def update_html(frame, images, im):
    """
    Updates the HTML display with the next frame of the animation.

    Parameters:
    - frame (int): Index of the frame to be displayed.
    - images (list of numpy.ndarray): List of spin field images representing the animation frames.
    - im (matplotlib.image.AxesImage): The image object to be updated.

    Returns:
    - matplotlib.image.AxesImage: Updated image object.
    """
    im.set_array(display_spin_field_html(images[frame]))
    return im,

def display_ising_sequence_html(images, folder_file_name="my_frame/animation.html"): 
    """
    Displays an Ising model simulation animation as an HTML file.

    Parameters:
    - images (list of numpy.ndarray): List of spin field images representing different simulation steps.
    - folder_file_name (str): The filename (with path) for the HTML animation.

    Returns:
    - matplotlib.animation.FuncAnimation: Animation object.
    """
    folder_file_name = convert_to_extension(folder_file_name, 'html')
    fig, ax = plt.subplots()
    im = ax.imshow(display_spin_field_html(images[0]), cmap='gray')
    ax.axis('off')
    anim = FuncAnimation(fig, update_html, frames=len(images), fargs=(images, im), interval=100)
    anim.save(folder_file_name, writer='html', fps=10)  # Save animation as HTML
    plt.close(fig)  # Close the figure to prevent it from displaying in the notebook
    return anim

def export_mp4(image_folder_name, output_file_name, frame_rate=24):
    """
    Export a sequence of PNG images as an MP4 video.

    Parameters:
    - image_folder_name (str): The directory containing PNG images to be used for video creation.
    - output_file_name (str): The name (with path) for the output MP4 video.
    - frame_rate (int): The frame rate (frames per second) of the output video. Default is 24.

    Returns:
    - bool: True if the video export is successful, False otherwise.
    """
    try: 
        output_file_name = convert_to_extension(output_file_name, 'mp4')

        # Directory containing your PNG images
        image_folder = image_folder_name
        # Output video file name
        video_name = output_file_name  # You can change the extension to .avi if needed

        # Get the list of image files
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort()  # Ensure images are in order

        # Create a list to store image paths
        image_paths = [os.path.join(image_folder, img) for img in images]

        # Read images and create the video using imageio
        with imageio.get_writer(video_name, fps=frame_rate) as writer:  # Adjust fps as needed
            for image_path in image_paths:
                writer.append_data(imageio.imread(image_path))
        return True
    except:
        return False