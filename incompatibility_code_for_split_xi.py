import xygame as xg
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
import multiprocessing


def find_eta_times_field(model, xi):
    lattice = model.lattice
    rows, cols = lattice.shape
    defect, anti_defect = xg.detect_defects(lattice)
    
    b_field = model.B_field_cartesian
    
    # Initialize a grid with the counters for each xi x xi square
    counters_defect = []
    counters_antidefect = []
    
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

            net_defect = counter_defect + counter_antidefect

            final_number_defect = (net_defect) * avg_magnitude
            
            # Append the magnitude to the result list
            if net_defect > 0:
                counters_defect.append(final_number_defect)
            if net_defect < 0:
                counters_antidefect.append(final_number_defect)

        if len(counters_defect) == 0 and len(counters_antidefect) == 0:
            return np.array([[0,0]])
        if len(counters_defect) == 0 and len(counters_antidefect) != 0:
            return np.array([[0, sum(counters_antidefect) / len(counters_antidefect)]])
        if len(counters_defect) != 0 and len(counters_antidefect) == 0:
            return np.array([[sum(counters_defect) / len(counters_defect), 0]])
        if len(counters_antidefect) != 0 and len(counters_defect) != 0:
            return np.array([[sum(counters_defect) / len(counters_defect), sum(counters_antidefect) / len(counters_antidefect)]]) 
    
    return np.array([[sum(counters_defect) / len(counters_defect), sum(counters_antidefect) / len(counters_antidefect)]])


# -------------------------------------------------------------------------------------------------------------

def run_simulation(b):
    rows = 100
    cols = 100
    beta = 0.9
    J = 1
    
    folder_name = f"split xi b is {b}"
    os.makedirs(folder_name, exist_ok=True)  # Create directory for storing results

    lattice = xg.lattice_generator(rows, cols)
    b_field = xg.field_generator(rows, cols, mag_left_most=0.0, mag_right_most=b, dir_left_most=0, dir_right_most=2*math.pi)
    model = xg.full_model(lattice=lattice, B_field=b_field, beta=beta, J=J)

    MCS = 10000
    time = MCS * rows * cols

    get_avg_energy = [xg.get_avg_energy(model)]
    get_defect = [xg.find_num_of_defects(model)]
    eta_dict = {}
    distances = np.arange(max(rows, cols) // 2)
    beta_list = [0.9, 1.2]
    xi_list = list(range(2, 81))

    model.beta = beta
    eta_list_time_avg = np.array([])
    eta_dict[f'eta_list_average_for_beta_{beta}'] = eta_list_time_avg
    sum_counter = 0

    x_axis = [0]

    for beta in beta_list:
        coerr_list = []
        defect_list = []
        model.beta = beta
        print(f'Running for b = {b}, beta = {beta}', flush=True)

        for i in range(time):
            xg.metropolis(model)
            if i % (100 * rows * cols) == 0:
                eta_list = np.empty((0, 2))
                for xi in xi_list:
                    eta_list = np.vstack((eta_list, find_eta_times_field(model, xi))) # add this
                x_axis.append(i)
                get_avg_energy.append(xg.get_avg_energy(model))
                get_defect.append(xg.find_num_of_defects(model))
                
                if eta_list_time_avg.size == 0:
                    eta_list_time_avg = eta_list
                else:
                    eta_list_time_avg += eta_list
                    sum_counter += 1

        eta_list_time_avg /= sum_counter
        eta_dict[f'eta_list_average_for_beta_{beta}'] = np.transpose(eta_list_time_avg)

        coerr = np.array(xg.correlation(model))
        defect_count = xg.find_num_of_defects(model)
        coerr_list.append(coerr)
        defect_list.append(defect_count)

        # Plot and save Scatter Plot for eta_list for defect
        plt.scatter(xi_list, eta_dict[f'eta_list_average_for_beta_{beta}'][0], color='green', marker='o', s=9, label='Data Points')
        plt.xlabel('Index')
        plt.ylabel('Eta List Average for defect')
        plt.title('Scatter Plot of Eta List Average')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(folder_name, f"scatter_plot_eta_for_defects_and_beta_{beta}.png"))  # Save plot
        plt.close()

        # Plot and save Scatter Plot for eta_list for antidefect
        plt.scatter(xi_list, eta_dict[f'eta_list_average_for_beta_{beta}'][1], color='green', marker='o', s=9, label='Data Points')
        plt.xlabel('Index')
        plt.ylabel('Eta List Average for anti-defect')
        plt.title('Scatter Plot of Eta List Average')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(folder_name, f"scatter_plot_eta_for_antidefect_and_beta_{beta}.png"))  # Save plot
        plt.close()

        # Save Coerr plot
        for i, coerr in enumerate(coerr_list):
            plt.plot(distances, coerr, label=f"Coerr {i+1}", marker='x')
        plt.xlabel("X Units")
        plt.ylabel("Coerr Values")
        plt.title(f"Coerr List vs X Units (b={b}, beta={beta})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(folder_name, f"coerr_list_for_beta_{beta}.png"))
        plt.close()

        # Save data as CSV
        data = {
            "Average Energy": get_avg_energy,
            "Defects": get_defect,
            "Eta List Average": eta_list_time_avg.tolist() if eta_list_time_avg.size > 0 else [],
            "Coerr List": [list(c) for c in coerr_list],
        }
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
        df.to_csv(os.path.join(folder_name, f"results_for_beta_{beta}.csv"), index=False)

    # Save Energy vs MCS plot
    plt.plot(x_axis, get_avg_energy, marker='o', label="Avg Energy", color='blue')
    plt.xlabel("Time (MCS)")
    plt.ylabel("Energy")
    plt.title(f"Energy vs MCS (b={b})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_name, "energy_vs_mcs.png"))
    plt.close()

    # Save Defect vs MCS plot
    plt.plot(x_axis, get_defect, marker='s', label="Defects", color='red')
    plt.xlabel("Time (MCS)")
    plt.ylabel("Defects")
    plt.title(f"Defects vs MCS (b={b})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_name, "defect_vs_mcs.png"))
    plt.close()

    print(f"Simulation for b = {b} completed. Results saved in '{folder_name}'.", flush=True)


if __name__ == "__main__":
    b_list = [2, 4, 6]

    # Use multiprocessing to run simulations for different values of b in parallel
    with multiprocessing.Pool(processes=len(b_list)) as pool:
        pool.map(run_simulation, b_list)

    print("All simulations completed.", flush=True)