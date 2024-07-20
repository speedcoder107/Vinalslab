import imageio
import numpy as np
from scipy.ndimage import convolve, generate_binary_structure
from scipy.constants import Boltzmann
import random
from ipywidgets import interact
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# constants
k_b = Boltzmann

class full_model():
    def __init__(self, lattice, B_field, beta   , net_energy=None, net_spin=None):
        
        '''
        Initializes an instance of the Ising model, providing a full picture of its configuration.

        Parameters: 
        - self (full_model): Reference to the instance of the class.
        - lattice (numpy.ndarray): Initial configuration of the lattice.
        - B_field (float): External magnetic field strength applied to the lattice.
        - temp (float): Temperature of the lattice system.

        Returns:
        - None
        '''

        self.lattice = lattice  # Initialize the lattice configuration
        self.B_field = B_field  # Initialize the external magnetic field strength
        self.beta = beta        # Initialize the temperature of the lattice system
        self.net_energy = net_energy
        self.net_spin = net_spin

        if net_energy == None:           
            net_energy = set_net_energy(self)
        
        if net_spin == None:
            net_spin = set_net_spin(self)

    
def field_generator(row,col,*args):

    '''
    Generates a field with 'row' number of rows and 'col' number of columns. The optional value, fixes the values of the field

    Parameters:
    - row (int): number of rows of the field.
    - y_column (int): number of columns of the field.
    - *args (int): fixes the values of all elements of the field to the given valud
    
    Returns:
    - 2D array of the field.
    '''

    if args == ():
        return np.random.choice([1,-1], size = (row, col)) # generates a random field with 1s and -1s if no arguement is provided
    return np.random.choice(args,size=(row,col))           # gererates a random field with all the arguments provided as elements



def set_net_energy(model):
    
    '''
    Calculates the energy of the lattice and stores it inside the full_model class of the given parameter.

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

        # convolves the field with the following matrix (kern):
        # [[False, True, False],
        #  [True, False, True],
        #  [False, True, False]]
        # I got this idea from the above source

        arr = -model.lattice * convolve(model.lattice, kern, mode='constant', cval=0) - model.lattice * model.B_field
        model.net_energy = arr.sum()
    except:
        raise Exception ("Cannot set net energy")


def set_net_spin(model):

    """
    Calculates the net spin of the lattice and stores it inside the full_model class of the given parameter.

    Parameters:
    - model (full_model): The entire ising model with all the information

    Returns:
    - None

    Raises:
    Exception: if it is unable to set energy
    """
    
    try:
        lattice = model.lattice
        model.net_spin =  np.sum(lattice)
    except:
        raise Exception ("cannot set net spin")
    
def get_net_energy(model):
    """
    Retrieves the net energy of the Ising model.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.

    Returns:
    - float: The net energy of the Ising model.
    """
    return model.net_energy

def get_net_spin(model):
    """
    Retrieves the net spin of the lattice in the Ising model.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.

    Returns:
    - float: The net spin of the lattice in the Ising model.
    """
    return model.net_spin

def get_lattice(model):
    """
    Retrieves the lattice of spins in the Ising model.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.

    Returns:
    - array-like: The lattice of spins in the Ising model.
    """
    return model.lattice

def get_B_field(model):
    """
    Retrieves the external magnetic field strength applied to the Ising model.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.

    Returns:
    - float: The strength of the external magnetic field applied to the Ising model.
    """
    return model.B_field

def get_temp(model):
    """
    Retrieves the temperature of the Ising model.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.

    Returns:
    - float: The temperature of the Ising model.
    """
    return 1 / (model.beta * k_b)

def get_neighborhood(lattice, row, col):
    '''
    Extracts the neighborhood of an element in a 2D NumPy array.

    Parameters:
    - lattice (2D array): Input 2D NumPy array representing the lattice.
    - n (int): Row index of the element whose neighborhood is to be extracted.
    - m (int): Column index of the element whose neighborhood is to be extracted.

    Returns:
    - 2D array: Array containing the neighborhood elements surrounding the specified element.
    '''

    rows, cols = lattice.shape
    
    # Define the indices of the neighborhood elements
    row_indices = [row-1, row, row+1]
    col_indices = [col-1, col, col+1]
    
    # Initialize the neighborhood array
    neighborhood = np.empty((3, 3), dtype=lattice.dtype)
    
    # Iterate over the indices and extract neighborhood elements
    for i, row_index in enumerate(row_indices):
        for j, col_index in enumerate(col_indices):
            # Ensure the indices are within the bounds of the lattice
            if 0 <= row_index < rows and 0 <= col_index < cols:
                # If indices are within bounds, extract the corresponding element
                neighborhood[i, j] = lattice[row_index, col_index]
            else:
                # If indices are out of bounds, use 0.
                neighborhood[i, j] = 0

    return neighborhood

def get_energy(model, row, col):
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

    # Get the lattice from the model
    lattice = model.lattice
    
    # Get the neighborhood of the spin at the specified position
    l_neighbor = get_neighborhood(lattice, row, col)
    
    # Get the value of the spin at the specified position
    elem = lattice[row, col]
    
    # Generate a 3x3 binary structure for convolution
    kern = generate_binary_structure(2, 1) 
    kern[1][1] = False
    
    # Compute the energy contribution from interactions with neighboring spins and external magnetic field
    # convolves the field with the following matrix (kern):
        # [[False, True, False],
        #  [True, False, True],
        #  [False, True, False]]
        # I got this idea from the above source
    
    energy = -elem * convolve(l_neighbor, kern, mode='constant', cval=0)[1, 1] - elem * model.B_field[row, col]
    
    return energy

def get_spin(model, row, col):
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


def metropolis(model):
    """
    Performs a single Metropolis step in the Ising model simulation.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.

    Returns:
    - bool: True if the spin was flipped, False otherwise.
    """

    # Get the lattice from the model
    lattice = model.lattice
    
    # Select a random position (row and column) in the lattice
    row = random.randint(0, lattice.shape[0] - 1)
    col = random.randint(0, lattice.shape[1] - 1)
    
    # Get the initial spin value at the selected position
    spin_i = get_spin(model, row, col)
    
    # Flip the spin to its opposite value
    spin_f = -spin_i
    
    # Calculate the initial and final energies at the selected position
    energy_i = get_energy(model, row, col)
    energy_f = -energy_i
    
    # Calculate the change in energy and spin
    d_energy = energy_f - energy_i
    d_spin = spin_f - spin_i
    
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
    
def my_algorithm(model, row, col):
    """
    Performs a single Metropolis step in the Ising model simulation.

    Parameters:
    - model (full_model): The Ising model object containing all relevant information.

    Returns:
    - bool: True if the spin was flipped, False otherwise.
    """

    # Get the lattice from the model
    lattice = model.lattice
    
    # Get the initial spin value at the selected position
    spin_i = get_spin(model, row, col)
    
    # Flip the spin to its opposite value
    spin_f = -spin_i
    
    # Calculate the initial and final energies at the selected position
    energy_i = get_energy(model, row, col)
    energy_f = -energy_i
    
    # Calculate the change in energy and spin
    d_energy = energy_f - energy_i
    d_spin = spin_f - spin_i
    
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

    # Define a function to display the images interactively
    def _show(frame=(0, len(images) - 1)):
        plt.figure(figsize=(6, 6))
        plt.imshow(display_spin_field(images[frame]), cmap='gray')
        plt.axis('off')
        plt.show()
    
    # Create an interactive widget to browse through the images
    return interact(_show)


def graph_list(data, x_label, y_label, title,):
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

    # Check if the extension is already the desired extension
    if ext == '.' + extension:
        return filename  # Nothing to do, return the original filename

    # Replace the current extension with the desired extension
    new_filename = name + '.' + extension

    # Rename the file if it exists
    if os.path.exists(filename):
        os.rename(filename, new_filename)
        print(f"File '{filename}' renamed to '{new_filename}'")
    
    return new_filename

# code for html rendering
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
    # Convert the folder_file_name to have the .html extension
    folder_file_name = convert_to_extension(folder_file_name, 'html')

    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    im = ax.imshow(display_spin_field_html(images[0]), cmap='gray')
    ax.axis('off')

    # Create the animation
    anim = FuncAnimation(fig, update_html, frames=len(images), fargs=(images, im), interval=100)
    
    # Save the animation as an HTML file
    anim.save(folder_file_name, writer='html', fps=10)
    
    # Close the figure to prevent it from displaying in the notebook
    plt.close(fig)
    
    return anim

# code for exporting video from animation frames
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
        # Convert the output file name to have the .mp4 extension
        output_file_name = convert_to_extension(output_file_name, 'mp4')

        # Get the list of PNG image files in the specified folder
        images = [img for img in os.listdir(image_folder_name) if img.endswith(".png")]
        images.sort()  # Ensure images are in order

        # Create a list to store the paths of image files
        image_paths = [os.path.join(image_folder_name, img) for img in images]

        # Create the MP4 video using imageio
        with imageio.get_writer(output_file_name, fps=frame_rate) as writer:
            for image_path in image_paths:
                writer.append_data(imageio.imread(image_path))
        
        # Return True if the video export is successful
        return True
    except:
        # Return False if an error occurs during the video export
        return False
