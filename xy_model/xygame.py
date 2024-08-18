# initialize and change the system between 0 and 2 pi.
import imageio
import numpy as np
import math
from scipy.ndimage import convolve, generate_binary_structure
from scipy.constants import Boltzmann, eV
import random
from ipywidgets import interact
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# constants
k_b = 1

class full_model():
    def __init__(self, lattice, B_field, beta, time, J, net_energy=None, net_spin=None):
        '''
        Initializes an instance of the Ising model, providing a full picture of its configuration.

        Parameters: 
        - self (full_model): Reference to the instance of the class.
        - lattice (numpy.ndlattice): Initial configuration of the lattice.
        - B_field (numpy.ndlattice): External magnetic field strength applied to the lattice.
        - temp (float): Temperature of the lattice system.
        - time (float): Simulation time or number of samples

        Returns:
        - None
        '''
        self.lattice = lattice  # Initialize the lattice configuration
        self.B_field = B_field  # Initialize the external magnetic field strength
        self.beta = beta        # Initialize the temperature of the lattice system
        self.time = time        # Initialize the simulation time or number of durations
        self.rows, self.cols = lattice.shape
        self.net_energy = net_energy
        self.net_spin = net_spin
        self.J = J

        if net_energy == None:           
            net_energy = set_net_energy(self)
        
        if net_spin == None:
            net_spin = set_net_spin(self)

def field_generator(row, col, left_most = 0.0, right_most = 2*math.pi):
    '''
    Generates a field with 'row' number of rows and 'col' number of columns. The optional value, fixes the values of the field

    Parameters:
    - row (int): number of rows of the field.
    - y_column (int): number of columns of the field.
    - *args (int): fixes the values of all elements of the field to the given valud
    
    Returns:
    - 2D lattice of the field.
    '''
    return np.random.uniform(left_most, right_most, size = (row, col)) # generates a random field with 0 and 2pi if no arguement is provided

def set_net_energy(full_model):
    """
    Compute the net net_energy of a 2D XY model configuration.

    Parameters:
    lattice (2D lattice): 2D lattice of angles in radians.

    Returns:
    float: Net net_energy of the configuration.
    """
    rows, cols = full_model.rows, full_model.cols
    lattice = full_model.lattice
    b_field = full_model.B_field
    net_energy = 0.0

    # Iterate over the lattice
    for i in range(rows):
        for j in range(cols):
            theta_ij = lattice[i, j]
            
            if (j + 1) < cols:
                # Right neighbor (with periodic boundary condition) 
                theta_right = lattice[i, (j + 1)]
                net_energy -= np.cos(theta_ij - theta_right)

            if (j - 1) >= 0:
                # left neighbor (with periodic boundary condition) 
                theta_left = lattice[i, (j - 1)]
                net_energy -=np.cos(theta_ij - theta_left)

            if (i + 1) < cols:
                # Bottom neighbor (with periodic boundary condition)
                theta_down = lattice[(i + 1), j]
                net_energy -= np.cos(theta_ij - theta_down)

            if (i - 1) >= 0:
                # up neighbor (with periodic boundary condition)
                theta_up = lattice[(i - 1), j]
                net_energy -= np.cos(theta_ij - theta_up)
            
            net_energy -= np.cos(theta_ij) * b_field[i,j] # not sure about this part
    
    full_model.net_energy = net_energy
    return net_energy

def set_net_spin(model):
    lattice = model.lattice
    net_spin =  np.sum(lattice)
    model.net_spin = net_spin
    return net_spin
    
def get_net_energy(model):
    return model.net_energy

def get_avg_energy(model):
    row, col = model.lattice.shape
    return model.net_energy/(row*col)

def get_avg_M(model):
    lattice = model.lattice
    row, col = model.lattice.shape
    cos = np.sum(np.cos(lattice))
    sin = np.sum(np.sin(lattice))
    M = np.sqrt(cos**2 + sin**2)/(row*col)
    return M

def get_net_spin(model):
    return model.net_spin

def get_lattice(model):
    return model.lattice

def get_B_field(model):
    return model.B_field

def get_temp(model):
    return 1/(model.beta * k_b)

def get_energy(model, row, col):
    """
    Calculate the energy contribution of a specific lattice site in a 2D spin model.
    
    Parameters:
    model (object): An object containing the lattice (2D numpy lattice) and the magnetic field (2D numpy lattice).
    row (int): The row index of the lattice site.
    col (int): The column index of the lattice site.
    
    Returns:
    float: The net energy contribution of the specified lattice site.
    """

    lattice = model.lattice
    b_field = model.B_field
    rows = lattice.shape[0]
    cols = lattice.shape[1]
    i = row
    j = col
    theta_ij = lattice[i, j]
    net_energy = 0.0

    
    if (j + 1) < cols:
        # Right neighbor (with periodic boundary condition) 
        theta_right = lattice[i, (j + 1)]
        net_energy -= np.cos(theta_ij - theta_right)

    if (j - 1) >= 0:
        # left neighbor (with periodic boundary condition) 
        theta_left = lattice[i, (j - 1)]
        net_energy -=np.cos(theta_ij - theta_left)

    if (i + 1) < rows:
        # Bottom neighbor (with periodic boundary condition)
        theta_down = lattice[(i + 1), j]
        net_energy -= np.cos(theta_ij - theta_down)

    if (i - 1) >= 0:
        # up neighbor (with periodic boundary condition)
        theta_up = lattice[(i - 1), j]
        net_energy -= np.cos(theta_ij - theta_up)
    
    net_energy -= np.cos(theta_ij) * b_field[i,j]  #not sure about this one

    return net_energy

def get_spin(model, row, col):
    return (model.lattice[row, col])

def metropolis(model):
    """
    Perform a single Metropolis update on the 2D lattice model.
    
    Parameters:
    model (object): An object containing the lattice (2D numpy lattice), temperature parameter beta,
                    and methods to get energy and spin of a site.
    
    Returns:
    bool: True if the update was accepted, False otherwise.
    """
    
    lattice = model.lattice
    row = random.randint(0, lattice.shape[0] - 1)
    col = random.randint(0, lattice.shape[0] - 1)
    
    old_value = lattice[row, col]
    spin_i = get_spin(model, row, col)
    
    dtheta = 0.5
    new_value = (lattice[row, col] + dtheta) % (2 * math.pi)  # Ensure new_value is within 0 to 2*pi
    spin_f = new_value
    
    energy_i = get_energy(model, row, col)
    lattice[row, col] = new_value
    energy_f = get_energy(model, row, col)
    
    d_energy = energy_f - energy_i
    d_spin = spin_f - spin_i
    
    if d_energy < 0 or np.random.random() < np.exp(-model.beta * d_energy):
        model.net_energy += d_energy
        model.net_spin += d_spin   
        return True
    else: 
        lattice[row, col] = old_value
        return False


def curl_calculator(lattice_2D):
    """
    Calculate the curl of a 2D gradient field.
    
    Parameters:
    gradient (np.ndlattice): A 2D numpy lattice where each element is a tuple (grad_x, grad_y).
    
    Returns:
    curl (np.ndlattice): A 2D numpy lattice representing the curl of the field.
    """
    
    if lattice_2D.ndim != 2 or not isinstance(lattice_2D[0, 0], tuple):
        raise ValueError("The input lattice_2D must be a 2D lattice of tuples.")
    
    # Get the dimensions of the lattice_2D field
    rows, cols = lattice_2D.shape
    
    # Initialize the curl lattice
    curl = np.zeros((rows, cols), dtype=float)
    
    # Calculate the curl using finite differences
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            grad_x_y_plus = lattice_2D[i+1, j][0]
            grad_x_y_minus = lattice_2D[i-1, j][0]
            grad_y_x_plus = lattice_2D[i, j+1][1]
            grad_y_x_minus = lattice_2D[i, j-1][1]
            
            partial_grad_y_x = (grad_y_x_plus - grad_y_x_minus) / 2.0
            partial_grad_x_y = (grad_x_y_plus - grad_x_y_minus) / 2.0
            
            curl[i, j] = partial_grad_y_x - partial_grad_x_y
    
    # Handle the boundaries with one-sided differences
    # Top and bottom rows
    for j in range(cols):
        curl[0, j] = lattice_2D[1, j][1] - lattice_2D[0, j][1] - (lattice_2D[0, j][0] - lattice_2D[0, j][0])
        curl[-1, j] = lattice_2D[-1, j][1] - lattice_2D[-2, j][1] - (lattice_2D[-1, j][0] - lattice_2D[-1, j][0])
    
    # Left and right columns
    for i in range(rows):
        curl[i, 0] = lattice_2D[i, 1][1] - lattice_2D[i, 0][1] - (lattice_2D[i, 0][0] - lattice_2D[i, 0][0])
        curl[i, -1] = lattice_2D[i, -1][1] - lattice_2D[i, -2][1] - (lattice_2D[i, -1][0] - lattice_2D[i, -1][0])
    
    return curl

def gradient_calculator(lattice):
    """
    Calculate the gradient of a 2D discrete field and return it as a 2D lattice of tuples.
    
    Parameters:
    field (np.ndlattice): A 2D numpy lattice representing the field.
    
    Returns:
    gradient (np.ndlattice): A 2D numpy lattice where each element is a tuple (grad_x, grad_y).
    """

    # Ensure the field is a 2D numpy lattice
    if not isinstance(lattice, np.ndarray) or lattice.ndim != 2:
        raise ValueError("The input lattice must be a 2D numpy array.")
        
    # Get the dimensions of the field
    rows, cols = lattice.shape
    
    # Initialize the lattice_2D lattices
    grad_x = np.zeros_like(lattice, dtype=float)
    grad_y = np.zeros_like(lattice, dtype=float)
    
    # Calculate the lattice_2D using central differences for interior points
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            grad_x[i, j] = (lattice[i, j+1] - lattice[i, j-1]) / 2.0
            grad_y[i, j] = (lattice[i+1, j] - lattice[i-1, j]) / 2.0
    
    # Handle the boundaries with one-sided differences
    # Top and bottom rows
    for j in range(cols):
        grad_y[0, j] = lattice[1, j] - lattice[0, j]
        grad_y[-1, j] = lattice[-1, j] - lattice[-2, j]
    
    # Left and right columns
    for i in range(rows):
        grad_x[i, 0] = lattice[i, 1] - lattice[i, 0]
        grad_x[i, -1] = lattice[i, -1] - lattice[i, -2]
    
    # Create a 2D lattice of tuples
    lattice_2D = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            lattice_2D[i, j] = (grad_x[i, j], grad_y[i, j])    
    return lattice_2D

def compute_angle_difference(angle1, angle2):
    """
    Compute the angle difference considering periodic boundary conditions.

    Parameters:
    angle1 (float): The first angle.
    angle2 (float): The second angle.

    Returns:
    float: The angle difference.
    """
    diff = angle1 - angle2
    return np.arctan2(np.sin(diff), np.cos(diff))


def detect_defects(lattice):
    """
    Detect vortices and antivortices in a 2D lattice of angles.

    Parameters:
    spins (np.ndlattice): A 2D numpy lattice of angles.

    Returns:
    list: A list of tuples indicating the positions of vortices and antivortices.
    """
    vortices = []
    Lx, Ly = lattice.shape
    
    for i in range(Lx - 1):
        for j in range(Ly - 1):
            # Angles at the corners of the plaquette
            theta_00 = lattice[i, j]
            theta_10 = lattice[i + 1, j]
            theta_11 = lattice[i + 1, j + 1]
            theta_01 = lattice[i, j + 1]

            # Compute the angle differences around the plaquette
            delta_theta1 = compute_angle_difference(theta_01, theta_00)
            delta_theta2 = compute_angle_difference(theta_11, theta_01)
            delta_theta3 = compute_angle_difference(theta_10, theta_11)
            delta_theta4 = compute_angle_difference(theta_00, theta_10)

            # Sum of the angle differences
            circulation = delta_theta1 + delta_theta2 + delta_theta3 + delta_theta4

            # Check if there's a vortex or antivortex
            if circulation > np.pi:
                vortices.append((j+ 0.5 , i +0.5))  # Vortex
            elif circulation < -np.pi:
                vortices.append((j+ 0.5 , i +0.5))  # Antivortex
    return vortices

def detect_defects_part_2(lattice):
    grad = gradient_calculator(lattice)
    curl = curl_calculator(grad)
    vortices = []
    for i in range(curl.shape[0]):
        for j in range(curl.shape[1]):
            if curl[i, j] > 2*np.pi: 
                vortices.append((i+ 0.5, j + 0.5))
            elif curl[i, j] < -2*np.pi:
                vortices.append((i + 0.5, j + 0.5))
    return vortices

def display_spin_field(field):
    """
    Convert a spin field to an 8-bit grayscale image for visualization.
    
    Parameters:
    field (np.ndlattice): A 2D numpy lattice representing the spin field with values in the range [-1, 1].
    
    Returns:
    np.ndlattice: A 2D numpy lattice representing the grayscale image.
    """
    image = np.uint8((field + 1) * 0.5 * 255)
    return image

def display_xy_sequence(images):
    """
    Display a sequence of 2D lattices as an interactive image slider using matplotlib.
    
    Parameters:
    images (list of np.ndlattice): A list of 2D numpy lattices to display as frames in the slider.
    
    Returns:
    IPython.display.display: An interactive slider to visualize the sequence of images.
    """
    def _show(frame=(0, len(images) - 1)):
        plt.figure(figsize=(6, 6))
        lattice = images[frame]
        plt.imshow(lattice, cmap='hsv', vmin=0, vmax=2*np.pi)
        plt.colorbar(label='Value')
        plt.title(f"Frame {frame}")
        plt.axis('off')
        plt.show()    
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
    plt.plot(x_values, data, marker='.', linestyle='-')
    
    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # Show the plot
    plt.grid(True)
    plt.show()

def convert_to_extension(filename, extension):
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
    return new_filename

# code for html rendering
def display_spin_field_html(field):
    image = np.uint8((field + 1) * 0.5 * 255)
    return image

def update_html(frame, images, im):
    im.set_array(display_spin_field_html(images[frame]))
    return im,

def display_xy_sequence_html(images, folder_file_name="my_frame/animation.html"): 
    folder_file_name = convert_to_extension(folder_file_name, 'html')
    fig, ax = plt.subplots()
    im = ax.imshow(display_spin_field_html(images[0]), cmap='hsv')
    ax.axis('off')
    anim = FuncAnimation(fig, update_html, frames=len(images), fargs=(images, im), interval=100)
    anim.save(folder_file_name, writer='html', fps=10)  # Save animation as HTML
    plt.close(fig)  # Close the figure to prevent it from displaying in the notebook
    return anim

# code for exporting video from animation frames
def export_mp4(image_folder_name, output_file_name, frame_rate = 24):
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
