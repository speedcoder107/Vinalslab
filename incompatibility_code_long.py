import xygame as xg
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

def find_eta_times_field(model, xi):
    lattice = model.lattice
    rows, cols = lattice.shape
    defect, anti_defect = xg.detect_defects(lattice)
    
    b_field = model.B_field_cartesian
    
    # Initialize a grid with the counters for each xi x xi square
    counters = []
    
    # Define the top-left corners of the squares
    for i in range(0, rows, xi):
        for j in range(0, cols, xi):
            # Define the boundaries of the current square
            x_min, x_max = i, i + xi
            y_min, y_max = j, j + xi
            
            # Count how many points fall into the current square
            counter_defect = sum(1 for x, y in defect if x_min <= x < x_max and y_min <= y < y_max)
            counter_antidefect = sum(-1 for x, y in anti_defect if x_min <= x < x_max and y_min <= y < y_max)

            #now, we get the magnetic field
            # Extract the xi x xi sub-region
            sub_matrix = b_field[i:i+xi, j:j+xi]
            
            # Flatten the sub-matrix to process all elements
            flattened = sub_matrix.reshape(-1, 2)
            
            # Compute averages of x and y components
            avg_x = np.mean(flattened[:, 0])
            avg_y = np.mean(flattened[:, 1])
            
            # Calculate the magnitude of the average vector
            avg_magnitude = float(np.sqrt(avg_x**2 + avg_y**2))

            final_number = (counter_defect + counter_antidefect) * avg_magnitude
            
            # Append the magnitude to the result list
            counters.append(final_number)

    # Compute and return the average
    return sum(counters) / len(counters) if counters else 0


# ---------------------------------------------------------------------

rows = 100
cols = 100
beta = 0.9
J = 1
b = 2

lattice = xg.lattice_generator(rows, cols)
b_field = xg.field_generator(rows, cols, mag_left_most=0.0, mag_right_most=b, dir_left_most=0, dir_right_most=2*math.pi)
model = xg.full_model(lattice=lattice, B_field=b_field, beta=beta, J=J)

MCS = 100
time = MCS * rows * cols

get_avg_energy = [xg.get_avg_energy(model)]
get_defect = [xg.find_num_of_defects(model)]
coerr_list = []
defect_list = []
eta_dict = {}
distances = np.arange(max(rows, cols) // 2)
xi_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

model.beta = beta
eta_list_time_avg = np.array([])
eta_dict[f'eta_list_average_for_beta_{beta}'] = eta_list_time_avg
sum_counter = 0

# Initialize the step size for the x-axis for graph only
step_size = 10 * rows * cols
x_axis = [0]  # Initialize x-axis for time steps

for i in range(time):
        xg.metropolis(model)
        if (i % 10* rows*cols) == 0:
            eta_list = np.array([])
            for xi in xi_list:
                eta_list = np.append(eta_list, find_eta_times_field(model, xi))
            x_axis.append(i)
            get_avg_energy.append(xg.get_avg_energy(model))
            get_defect.append(xg.find_num_of_defects(model))
            if np.array(eta_list_time_avg).size == 0:
                eta_list_time_avg = eta_list
            else:
                eta_list_time_avg = eta_list_time_avg + eta_list
                sum_counter += 1

eta_list_time_avg = eta_list_time_avg / sum_counter
eta_dict[f'eta_list_average_for_beta_{beta}'] = eta_list_time_avg

coerr = np.array(xg.correlation(model))
defect_count = xg.find_num_of_defects(model)
coerr_list.append(coerr)
defect_list.append(defect_count)

# Plot and save Energy vs MCS
plt.plot(x_axis, get_avg_energy, marker='o', label="Avg Energy", color='blue')
plt.xlabel("Time (MCS)")
plt.ylabel("Energy")
plt.title("Energy vs MCS")
plt.legend()
plt.grid(True)
plt.savefig("energy_vs_mcs.png")  # Save plot
plt.close()

# Plot and save Defect vs MCS
plt.plot(x_axis, get_defect, marker='s', label="Defects", color='red')
plt.xlabel("Time (MCS)")
plt.ylabel("Defects")
plt.title("Defects vs MCS")
plt.legend()
plt.grid(True)
plt.savefig("defect_vs_mcs.png")  # Save plot
plt.close()

# Plot and save Scatter Plot for eta_list
plt.scatter(xi_list, eta_dict['eta_list_average_for_beta_0.9'], color='green', marker='o', s=9, label='Data Points')
plt.xlabel('Index')
plt.ylabel('Eta List Average')
plt.title('Scatter Plot of Eta List Average')
plt.legend()
plt.grid(True)
plt.savefig("scatter_plot_eta.png")  # Save plot
plt.close()

# Plot and save Coerr List
for i, coerr in enumerate(coerr_list):
    plt.plot(distances, coerr, label=f"Coerr {i+1}", marker='x')
plt.xlabel("X Units")
plt.ylabel("Coerr Values")
plt.title("Coerr List vs X Units")
plt.legend()
plt.grid(True)
plt.savefig("coerr_list.png")  # Save plot
plt.close()

# Save all lists and results to a CSV file
data = {
    "Average Energy": get_avg_energy,
    "Defects": get_defect,
    "Eta List Average": eta_list_time_avg.tolist() if eta_list_time_avg.size > 0 else [],
    "Coerr List": [list(c) for c in coerr_list],
}
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
df.to_csv("results.csv", index=False)

print("Plots saved as image files and data saved to results.csv.")
