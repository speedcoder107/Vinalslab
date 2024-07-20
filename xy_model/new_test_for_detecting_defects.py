import numpy as np
import matplotlib.pyplot as plt

def curl_calculator(lattice_2D):
    lattice_2D = np.asarray(lattice_2D)
    if lattice_2D.ndim != 2 or not isinstance(lattice_2D[0, 0], tuple):
        raise ValueError("The input lattice_2D must be a 2D array of tuples.")
    
    # Get the dimensions of the lattice_2D field
    rows, cols = lattice_2D.shape
    
    # Initialize the curl array
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
    # Ensure the field is a 2D numpy array
    lattice = np.asarray(lattice)
    if lattice.ndim != 2:
        raise ValueError("The input field must be a 2D array.")
    
    # Get the dimensions of the field
    rows, cols = lattice.shape
    
    # Initialize the lattice_2D arrays
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
    
    # Create a 2D array of tuples
    lattice_2D = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            lattice_2D[i, j] = (grad_x[i, j], grad_y[i, j])    
    return lattice_2D

def find_extreme_points(array):
    """Returns a list of tuples (x, y) of all indexes in the array 
    where the values are greater than 2pi or less than -2pi.
    
    Parameters:
        array (np.ndarray): A 2D numpy array.
        
    Returns:
        List[Tuple[int, int]]: List of tuples with the indexes of the extreme points.
    """
    extreme_points = []
    threshold = np.pi
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > threshold or array[i, j] < -threshold:
                extreme_points.append((i, j))
    return extreme_points


Lx, Ly = 20, 20
x = np.linspace(-1, 1, Lx)
y = np.linspace(-1, 1, Ly)
X, Y = np.meshgrid(x, y)


# Compute the angles for curl (rotational pattern)
angles = np.arctan2(Y, X) + np.pi / 2

# Detect defects
grad = gradient_calculator(angles)
curl = curl_calculator(grad)
with open('output.txt', 'w') as f:
    for item in curl:

        f.write("%s\n" % item)
defects = find_extreme_points(curl)
print(defects)

# Plot the spin structure and the detected defects
X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly))
U = np.cos(angles)
V = np.sin(angles)

plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, pivot='mid')
for (x, y) in defects:
    plt.plot(x, y, 'ro' if x%2==0 else 'bo')  # red for vortices, blue for antivortices
plt.xlim(-1, Lx)
plt.ylim(-1, Ly)
plt.title('Spin Structure with Detected Defects')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.show()

