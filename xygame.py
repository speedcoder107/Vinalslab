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
    def __init__(self, lattice, B_field, beta, J, uniform_B_field = 0, net_energy=None, net_spin=None, B_field_cartesian = None, time = None):
        '''
        Initializes an instance of the Ising model, providing a full picture of its configuration.

        Parameters: 
        - self (full_model): Reference to the instance of the class.
        - lattice (numpy.ndlattice): Initial configuration of the lattice.
        - B_field (numpy.ndlattice): External magnetic field strength applied to the lattice.
        - temp (float): Temperature of the lattice system.

        Returns:
        - None
        '''
        self.lattice = lattice  # Initialize the lattice configuration
        self.B_field = B_field  # Initialize the external magnetic field strength
        self.uniform_B_field = uniform_B_field
        self.beta = beta        # Initialize the temperature of the lattice system
        self.rows, self.cols = lattice.shape
        self.net_energy = net_energy
        self.net_spin = net_spin
        self.J = J
        self.time = time

        rows, cols, _ = B_field.shape

        B_field_cartesian = np.zeros((rows, cols, 2))

        # Convert to x and y components
        for i in range(rows):
            for j in range(cols):
                magnitude = B_field[i, j, 0]
                direction = B_field[i, j, 1]
                
                # Calculate x and y components
                x = magnitude * np.cos(direction)
                y = magnitude * np.sin(direction)
                
                # Store the components in the new array
                B_field_cartesian[i, j] = [x, y]
        
        self.B_field_cartesian = B_field_cartesian


        if net_energy == None:           
            net_energy = set_net_energy(self)
        
        if net_spin == None:
            net_spin = set_net_spin(self)

def lattice_generator(row, col, left_most = 0.0, right_most = 2*math.pi):
    '''
    Generates a lattice with 'row' number of rows and 'col' number of columns. The optional value, fixes the values of the field

    Parameters:
    - row (int): number of rows of the field.
    - y_column (int): number of columns of the field.
    - *args (int): fixes the values of all elements of the field to the given valud
    
    Returns:
    - 2D lattice of the field.
    '''
    return np.random.uniform(left_most, right_most, size = (row, col)) # generates a random field with 0 and 2pi if no arguement is provided

def field_generator(row, col, mag_left_most = 0.0, mag_right_most = 1, dir_left_most = 0, dir_right_most = 2*math.pi):
    '''
    Generates a field with 'row' number of rows and 'col' number of columns. The optional value, fixes the values of the field

    Parameters:
    - row (int): number of rows of the field.
    - y_column (int): number of columns of the field.
    - *args (int): fixes the values of all elements of the field to the given valud
    
    Returns:
    - 2D lattice of the field.
    '''
    # Initialize the 3D array with zeros
    field = np.zeros((row, col, 2))
    
    # Iterate over each position in the array
    for i in range(row):
        for j in range(col):
            # Generate random magnitude within the specified range
            magnitude = np.random.uniform(mag_left_most, mag_right_most)
            # Generate random direction within the specified range
            direction = np.random.uniform(dir_left_most, dir_right_most)
            # Assign the magnitude and direction to the current position in the array
            field[i, j] = [magnitude, direction]
    
    return field

import numpy as np
import matplotlib.pyplot as plt

def picture(model, title=None):
    """
    Generates and saves a graphical picture of the spin lattice and overlaid magnetic field,
    along with defect detection markers.
    
    Parameters:
    - model: An instance of full_model which contains:
        - lattice: 2D numpy array of spin angles.
        - B_field_cartesian: 3D numpy array with shape (rows, cols, 2) representing the x and y
          components of the magnetic field.
    
    Process:
    1. Creates a meshgrid corresponding to the lattice dimensions.
    2. Computes the x and y components (cos and sin) of the lattice spins.
    3. Plots the spin lattice as arrows using plt.quiver.
    4. Overlays the magnetic field arrows in blue with low alpha for transparency.
    5. Detects defects via xg.detect_defects on the spin lattice and plots them:
       - Defects are marked with red circles.
       - Anti-defects are marked with blue circles.
    6. Sets axis limits, labels, title, and grid.
    7. Saves the image to disk and displays the plot.
    """
    
    # Get lattice dimensions
    rows, cols = model.lattice.shape
    
    # Create meshgrid for the lattice (x corresponds to column index, y to row index)
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Compute arrow components for the spin lattice (each element in model.lattice is an angle)
    U = np.cos(model.lattice)
    V = np.sin(model.lattice)
    
    # Start plotting
    plt.figure(figsize=(8, 8))
    
    # Plot the spin lattice arrows (using black for clear visibility)
    plt.quiver(X, Y, U, V, pivot='mid', color='k', label='Spins')
    
    # Overlay the magnetic field arrows
    # B_field_cartesian is assumed to have been computed in the model's __init__
    U_field = model.B_field_cartesian[:, :, 0]
    V_field = model.B_field_cartesian[:, :, 1]
    
    # Plot the B-field with blue arrows, lower opacity (alpha)
    plt.quiver(X, Y, U_field, V_field, pivot='mid', color='blue', alpha=0.3, label='B Field')
    
    # Detect defects on the spin lattice; xg.detect_defects is assumed available in your environment
    # i_lattice is the lattice of spin angles
    i_lattice = model.lattice
    defects, anti_defects = detect_defects(i_lattice)
    
    # Plot defects (vortices and antivortices)
    for (x, y) in defects:
        plt.plot(x, y, 'ro', markersize=5)  # red for vortices
    for (x, y) in anti_defects:
        plt.plot(x, y, 'bo', markersize=5)  # blue for antivortices
    
    # Set axis limits; adjust these if your lattice dimensions differ from 100x100
    plt.xlim(-1, cols)
    plt.ylim(-1, rows)
    
    time = model.time
    if time is not None:
        MCS = model.time/(rows*cols)
        plt.title('Spin Structure with Detected Defects and anti-defects at MCS = ' + str(MCS))
    else:    
        plt.title('Spin Structure with Detected Defects and anti-defects')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    
    # Save the figure
    plt.savefig(title, dpi=300)

    plt.show()    
    return True

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
            
            # Right neighbor (with periodic boundary condition) 
            theta_right = lattice[i, (j + 1)%cols]
            net_energy -= np.cos(theta_ij - theta_right)

            # left neighbor (with periodic boundary condition) 
            theta_left = lattice[i, (j - 1)%cols]
            net_energy -=np.cos(theta_ij - theta_left)

            # Bottom neighbor (with periodic boundary condition)
            theta_down = lattice[(i + 1)%rows, j]
            net_energy -= np.cos(theta_ij - theta_down)

            # up neighbor (with periodic boundary condition)
            theta_up = lattice[(i - 1)%rows, j]
            net_energy -= np.cos(theta_ij - theta_up)
            
            net_energy -= np.cos(theta_ij - b_field[i,j,1]) * b_field[i,j, 0] # energy due to magnetic field
    
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

    
    # Right neighbor (with periodic boundary condition) 
    theta_right = lattice[i, (j + 1)%cols]
    net_energy -= np.cos(theta_ij - theta_right)

    # left neighbor (with periodic boundary condition) 
    theta_left = lattice[i, (j - 1)%cols]
    net_energy -=np.cos(theta_ij - theta_left)

    # Bottom neighbor (with periodic boundary condition)
    theta_down = lattice[(i + 1)%rows, j]
    net_energy -= np.cos(theta_ij - theta_down)

    # up neighbor (with periodic boundary condition)
    theta_up = lattice[(i - 1)%rows, j]
    net_energy -= np.cos(theta_ij - theta_up)
    
    net_energy -= np.cos(theta_ij - b_field[i,j,1]) * b_field[i,j, 0] # energy due to magnetic field

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
    
    if random.random() > 0.5:
        dtheta = 0.1
    else:
        dtheta = -0.1

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


# def curl_calculator(lattice_2D):
#     """
#     Calculate the curl of a 2D gradient field.
    
#     Parameters:
#     gradient (np.ndlattice): A 2D numpy lattice where each element is a tuple (grad_x, grad_y).
    
#     Returns:
#     curl (np.ndlattice): A 2D numpy lattice representing the curl of the field.
#     """
    
#     if lattice_2D.ndim != 2 or not isinstance(lattice_2D[0, 0], tuple):
#         raise ValueError("The input lattice_2D must be a 2D lattice of tuples.")
    
#     # Get the dimensions of the lattice_2D field
#     rows, cols = lattice_2D.shape
    
#     # Initialize the curl lattice
#     curl = np.zeros((rows, cols), dtype=float)
    
#     # Calculate the curl using finite differences
#     for i in range(1, rows-1):
#         for j in range(1, cols-1):
#             grad_x_y_plus = lattice_2D[i+1, j][0]
#             grad_x_y_minus = lattice_2D[i-1, j][0]
#             grad_y_x_plus = lattice_2D[i, j+1][1]
#             grad_y_x_minus = lattice_2D[i, j-1][1]
            
#             partial_grad_y_x = (grad_y_x_plus - grad_y_x_minus) / 2.0
#             partial_grad_x_y = (grad_x_y_plus - grad_x_y_minus) / 2.0
            
#             curl[i, j] = partial_grad_y_x - partial_grad_x_y
    
#     # Handle the boundaries with one-sided differences
#     # Top and bottom rows
#     for j in range(cols):
#         curl[0, j] = lattice_2D[1, j][1] - lattice_2D[0, j][1] - (lattice_2D[0, j][0] - lattice_2D[0, j][0])
#         curl[-1, j] = lattice_2D[-1, j][1] - lattice_2D[-2, j][1] - (lattice_2D[-1, j][0] - lattice_2D[-1, j][0])
    
#     # Left and right columns
#     for i in range(rows):
#         curl[i, 0] = lattice_2D[i, 1][1] - lattice_2D[i, 0][1] - (lattice_2D[i, 0][0] - lattice_2D[i, 0][0])
#         curl[i, -1] = lattice_2D[i, -1][1] - lattice_2D[i, -2][1] - (lattice_2D[i, -1][0] - lattice_2D[i, -1][0])
    
#     return curl

# def gradient_calculator(lattice):
#     """
#     Calculate the gradient of a 2D discrete field and return it as a 2D lattice of tuples.
    
#     Parameters:
#     field (np.ndlattice): A 2D numpy lattice representing the field.
    
#     Returns:
#     gradient (np.ndlattice): A 2D numpy lattice where each element is a tuple (grad_x, grad_y).
#     """

#     # Ensure the field is a 2D numpy lattice
#     if not isinstance(lattice, np.ndarray) or lattice.ndim != 2:
#         raise ValueError("The input lattice must be a 2D numpy array.")
        
#     # Get the dimensions of the field
#     rows, cols = lattice.shape
    
#     # Initialize the lattice_2D lattices
#     grad_x = np.zeros_like(lattice, dtype=float)
#     grad_y = np.zeros_like(lattice, dtype=float)
    
#     # Calculate the lattice_2D using central differences for interior points
#     for i in range(1, rows-1):
#         for j in range(1, cols-1):
#             grad_x[i, j] = (lattice[i, j+1] - lattice[i, j-1]) / 2.0
#             grad_y[i, j] = (lattice[i+1, j] - lattice[i-1, j]) / 2.0
    
#     # Handle the boundaries with one-sided differences
#     # Top and bottom rows
#     for j in range(cols):
#         grad_y[0, j] = lattice[1, j] - lattice[0, j]
#         grad_y[-1, j] = lattice[-1, j] - lattice[-2, j]
    
#     # Left and right columns
#     for i in range(rows):
#         grad_x[i, 0] = lattice[i, 1] - lattice[i, 0]
#         grad_x[i, -1] = lattice[i, -1] - lattice[i, -2]
    
#     # Create a 2D lattice of tuples
#     lattice_2D = np.empty((rows, cols), dtype=object)
#     for i in range(rows):
#         for j in range(cols):
#             lattice_2D[i, j] = (grad_x[i, j], grad_y[i, j])    
#     return lattice_2D

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
    anti_vortices = []
    Lx, Ly = lattice.shape
    
    for i in range(Lx):
        for j in range(Ly):
            # Angles at the corners of the plaquette

            # Use modular indexing to wrap around
            theta_00 = lattice[i % lattice.shape[0], j % lattice.shape[1]]
            theta_10 = lattice[(i + 1) % lattice.shape[0], j % lattice.shape[1]]
            theta_11 = lattice[(i + 1) % lattice.shape[0], (j + 1) % lattice.shape[1]]
            theta_01 = lattice[i % lattice.shape[0], (j + 1) % lattice.shape[1]]

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
                anti_vortices.append((j+ 0.5 , i +0.5))  # Antivortex
    return vortices, anti_vortices

def find_num_of_defects(model):
    lattice = model.lattice
    defect, anti_defect = detect_defects(lattice)
    return (len(defect) + len(anti_defect))

# def detect_defects_part_2(lattice):
#     grad = gradient_calculator(lattice)
#     curl = curl_calculator(grad)
#     vortices = []
#     for i in range(curl.shape[0]):
#         for j in range(curl.shape[1]):
#             if curl[i, j] > 2*np.pi: 
#                 vortices.append((i+ 0.5, j + 0.5))
#             elif curl[i, j] < -2*np.pi:
#                 vortices.append((i + 0.5, j + 0.5))
#     return vortices

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


def correlation(model):
    """Compute the correlation function."""
    spins = model.lattice
    rows, cols = spins.shape
    corrs = []
    for r in range(max(rows, cols) // 2):
        corr = 0
        for i in range(max(rows, cols)):
            for j in range(max(rows, cols)):
                corr += np.cos(spins[i, j] - spins[(i+r) % max(rows, cols), j])
                corr += np.cos(spins[i, j] - spins[i, (j+r) % max(rows, cols)])
        corr /= (2 * max(rows, cols) * max(rows, cols))  # Average over both directions
        corrs.append(corr)
    return corrs

def get_neighbors(row, col, model):
    """
    Get the neighboring sites of a given (row, col) position on a 2D lattice 
    with periodic boundary conditions.
    
    Parameters:
    row (int): Row index of the lattice site.
    col (int): Column index of the lattice site.
    lattice_shape (tuple): Shape of the lattice (number of rows, number of columns).
    
    Returns:
    list of tuples: A list of neighboring (row, col) positions.
    """
    nrows, ncols = model.lattice.shape
    neighbors = []
    
    # Neighbor above (wraps to bottom if at the top)
    neighbors.append(((row - 1) % nrows, col))
    # Neighbor below (wraps to top if at the bottom)
    neighbors.append(((row + 1) % nrows, col))
    # Neighbor to the left (wraps to right if at the left edge)
    neighbors.append((row, (col - 1) % ncols))
    # Neighbor to the right (wraps to left if at the right edge)
    neighbors.append((row, (col + 1) % ncols))
    
    return neighbors

def effective_field(model, row, col):
    """
    Compute the local effective field H_eff at a given site.
    """
    neighbors = np.array([get_spin(model, i, j) for i, j in get_neighbors(row, col, model)])
    neighbors_sum = np.array([sum(np.cos(neighbors)) , sum(np.sin(neighbors))]) 
    h_local_mag, h_local_dir = model.B_field[row, col] 
    h_local_x = h_local_mag * np.cos(h_local_dir)
    h_local_y = h_local_mag * np.sin(h_local_dir)
    h_local = np.array([h_local_x, h_local_y])
    h_global = np.array([model.uniform_B_field, 0])
    H_eff = neighbors_sum + h_local + h_global
    return H_eff


def relax_step(model, alpha):
    """
    Perform a single relaxation update (sequential rotation or energy-conserving spin flip) 
    on the 2D lattice model based on the relaxation method from the paper.
    
    Parameters:
    model (object): An object containing the lattice (2D numpy array), temperature parameter beta,
                    and methods to get energy and spin of a site.
    alpha (float): Probability of performing a sequential rotation. Default is 0.8.
    
    Returns:
    None
    """
    lattice = model.lattice
    rows, cols = lattice.shape
    for row_seq in range(rows):
        for col_seq in range(cols):
            # Determine update type based on alpha
            if random.random() < alpha:
                old_value = lattice[row_seq, col_seq]
                spin_i = np.array([np.cos(get_spin(model, row_seq, col_seq)), np.sin(get_spin(model, row_seq, col_seq))])
                H_eff = effective_field(model, row_seq, col_seq)
                # Sequential rotation toward H_eff
                delta_theta = 0.1 if np.dot(spin_i, H_eff) < 0 else -0.1
                new_value = (old_value + delta_theta) % (2 * math.pi)
                energy_i = get_energy(model, row_seq, col_seq)
                lattice[row_seq, col_seq] = new_value
                energy_f = get_energy(model, row_seq, col_seq)

            else:
                row = random.randint(0, lattice.shape[0] - 1)
                col = random.randint(0, lattice.shape[0] - 1)
                H_eff = effective_field(model, row, col)
                # Energy-conserving spin flip
                spin_i = np.array([np.cos(get_spin(model, row, col)), np.sin(get_spin(model, row, col))])
                projection = 2 * np.dot(spin_i, H_eff) / np.linalg.norm(H_eff)**2
                new_spin = projection * H_eff - spin_i
                direction = math.atan2(new_spin[1], new_spin[0])
                # magnitude = math.sqrt(new_spin[0]**2 + new_spin[1]**2)                
                new_value = direction
                # print(magnitude)

                if col_seq == 0:
                    row_seq -= 1
                    col_seq = cols - 1
                else:
                    col_seq -= 1

                energy_i = get_energy(model, row, col)
                lattice[row, col] = new_value
                energy_f = get_energy(model, row, col)
            
            d_energy = energy_f - energy_i
            
            # Update net energy and net spin
            model.net_energy += d_energy
            model.net_spin += new_value - spin_i

def find_eta_dot_field(model, xi):
    lattice = model.lattice
    rows, cols = lattice.shape
    defect, anti_defect = detect_defects(lattice)
    
    b_field = model.B_field_cartesian
    
    # Initialize a grid with the counters for each xi x xi square
    avg_eta_dot_field_defect = np.empty((0, 2))  
    avg_eta_dot_field_antidefect = np.empty((0, 2))
    
    # Define the top-left corners of the squares
    for i in range(0, rows, xi-1):
        for j in range(0, cols, xi-1):
            # Define the boundaries of the current square
            x_min, x_max = i, i + xi - 1    # Adjusted to exclude the rightmost column
            y_min, y_max = j, j + xi - 1    # Adjusted to exclude the bottom row

            # Count how many points fall into the current square. This is my eta value, aka my number of unpariex defects and anti-defects
            num_unpaired_defects = 0
            for x, y in defect:
                if x_min <= x < x_max and y_min <= y < y_max:
                    num_unpaired_defects += 1

            num_unpaired_antidefect = 0
            for x, y in anti_defect:
                if x_min <= x < x_max and y_min <= y < y_max:
                    num_unpaired_antidefect -= 1

            #now, we get the magnetic field
            # Get the correct indices using modular arithmetic (wrap around)
            modulated_rows = [(y % b_field.shape[0]) for y in range(y_min, y_max + 1)]
            modulated_cols = [(x % b_field.shape[1]) for x in range(x_min, x_max + 1)]

            # Create the submatrix using advanced indexing
            sub_matrix = b_field[np.ix_(modulated_rows, modulated_cols)]            
            
            # Flatten the sub-matrix to process all elements
            flattened = sub_matrix.reshape(-1, 2)
            
            # Compute averages of x and y components
            avg_x = np.mean(flattened[:, 0])
            avg_y = np.mean(flattened[:, 1])

            avg_field = np.array([avg_x, avg_y])
    
            eta_dot_field_defect = (num_unpaired_defects) * avg_field
            eta_dot_field_anti_defect = (num_unpaired_antidefect) * avg_field
            
            # Append the magnitude to the result list
            avg_eta_dot_field_defect = np.vstack([avg_eta_dot_field_defect, eta_dot_field_defect])
            avg_eta_dot_field_antidefect = np.vstack([avg_eta_dot_field_antidefect, eta_dot_field_anti_defect])

    avg_eta_dot_field_defect = sum(avg_eta_dot_field_defect) / len(avg_eta_dot_field_defect)
    avg_eta_dot_field_antidefect = sum(avg_eta_dot_field_antidefect) / len(avg_eta_dot_field_antidefect)
            
    return np.array([avg_eta_dot_field_defect, avg_eta_dot_field_antidefect])

def find_eta(model, xi):
    lattice = model.lattice
    rows, cols = lattice.shape
    defect, anti_defect = detect_defects(lattice)
    
    # Initialize a grid with the counters for each xi x xi square
    avg_eta_defect = np.array([])
    avg_eta_antidefect = np.array([])
    
    # Define the top-left corners of the squares
    for i in range(0, rows, xi-1):
        for j in range(0, cols, xi-1):
            # Define the boundaries of the current square
            x_min, x_max = i, i + xi - 1    # Adjusted to exclude the rightmost column
            y_min, y_max = j, j + xi - 1    # Adjusted to exclude the bottom row

            # Count how many points fall into the current square. This is my eta value, aka my number of unpariex defects and anti-defects
            num_unpaired_defects = 0
            for x, y in defect:
                if x_min <= x < x_max and y_min <= y < y_max:
                    num_unpaired_defects += 1

            num_unpaired_antidefects = 0
            for x, y in anti_defect:
                if x_min <= x < x_max and y_min <= y < y_max:
                    num_unpaired_antidefects -= 1  
            
            # Append the magnitude to the result list
            avg_eta_defect = np.append(avg_eta_defect, num_unpaired_defects)
            avg_eta_antidefect = np.append(avg_eta_antidefect, num_unpaired_antidefects)

    avg_eta_defect = sum(avg_eta_defect) / len(avg_eta_defect)
    avg_eta_antidefect = sum(avg_eta_antidefect) / len(avg_eta_antidefect)
            
    return np.array([avg_eta_defect, avg_eta_antidefect])



def find_field(model, xi):
    lattice = model.lattice
    rows, cols = lattice.shape
    
    b_field = model.B_field_cartesian
    
    # Initialize a grid with the counters for each xi x xi square
    avg_field_for_all_squares = np.empty((0, 2))  

    
    # Define the top-left corners of the squares
    for i in range(0, rows, xi-1):
        for j in range(0, cols, xi-1):
            # Define the boundaries of the current square
            x_min, x_max = i, i + xi - 1    # Adjusted to exclude the rightmost column
            y_min, y_max = j, j + xi - 1    # Adjusted to exclude the bottom row

            #now, we get the magnetic field
            # Get the correct indices using modular arithmetic (wrap around)
            modulated_rows = [(y % b_field.shape[0]) for y in range(y_min, y_max + 1)]
            modulated_cols = [(x % b_field.shape[1]) for x in range(x_min, x_max + 1)]

            # Create the submatrix using advanced indexing
            sub_matrix = b_field[np.ix_(modulated_rows, modulated_cols)]            
            
            # Flatten the sub-matrix to process all elements
            flattened = sub_matrix.reshape(-1, 2)
            
            # Compute averages of x and y components
            avg_x = np.mean(flattened[:, 0])
            avg_y = np.mean(flattened[:, 1])

            avg_field_for_current_square = np.array([avg_x, avg_y])
            avg_field_for_all_squares = np.vstack([avg_field_for_all_squares, avg_field_for_current_square])
    avg_field_for_all_squares = sum(avg_field_for_all_squares)/len(avg_field_for_all_squares)

    return avg_field_for_all_squares