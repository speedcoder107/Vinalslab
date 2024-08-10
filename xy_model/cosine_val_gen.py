import numpy as np
import csv

# Number of points
num_points = 20

# Generate the theta values from 0 to 2*pi
theta_values = np.linspace(0, 2*np.pi, num_points)

# Calculate the cosine values
cosine_values = np.cos(theta_values)

# Create the CSV file
with open('cosine_values.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['Index', 'theta', 'cosine'])
    
    # Write the data
    for i, (theta, cosine) in enumerate(zip(theta_values, cosine_values)):
        writer.writerow([i, theta, cosine])

print("CSV file 'cosine_values.csv' created successfully.")
