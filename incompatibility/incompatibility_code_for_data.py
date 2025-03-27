import xygame as xg
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
import multiprocessing
import itertools

# -------------------------------------------------------------------------------------------------------------

def run_simulation(args):
    b, beta = args
    rows = 100
    cols = 100
    J = 1
    
    lattice = xg.lattice_generator(rows, cols)
    b_field = xg.field_generator(rows, cols, mag_left_most=0.0, mag_right_most=b, dir_left_most=0, dir_right_most=2*math.pi)
    model = xg.full_model(lattice=lattice, B_field=b_field, beta=beta, J=J, time = 0)

    MCS = 10000
    time = MCS * rows * cols

    xi_list = list(range(2, 81))
    avg_energy_list = np.array([])
    defect_list = np.array([])
    distances = np.arange(max(rows, cols) // 2).astype(float)
    eta_dot_field_list = np.empty((0, len(xi_list), 2, 2))
    eta_list = np.empty((0, len(xi_list), 2))
    b_field_list = np.empty((0, len(xi_list), 2))

    model.beta = beta

    # Initialize the step size for the x-axis for graph only
    time_axis = np.array([])  # Initialize x-axis for time steps

    for i in range (time):
        xg.metropolis(model)
        model.time += 1
    
    for i in range(time):
        if i % (100 *rows * cols) == 0:
            print(model.time)
            eta_dot_field = np.empty((0, 2, 2))
            eta = np.empty((0, 2))
            b_field = np.empty((0, 2))
            for xi in xi_list:
                eta_dot_field = np.append(eta_dot_field, [xg.find_eta_dot_field(model, xi)], axis=0)
                eta = np.append(eta, [xg.find_eta(model, xi)], axis=0)
                b_field = np.append(b_field, [xg.find_field(model, xi)], axis=0)
            eta_dot_field_list = np.append(eta_dot_field_list, [eta_dot_field], axis=0)
            eta_list = np.append(eta_list, [eta], axis=0)
            b_field_list = np.append(b_field_list, [b_field], axis=0)
            time_axis = np.append(time_axis, model.time/(rows*cols))  # Append time step to x-axis
            avg_energy_list = np.append(avg_energy_list, xg.get_avg_energy(model))
            defect_list = np.append(defect_list, xg.find_num_of_defects(model))
        xg.metropolis(model)
        model.time += 1

    coerr_list = np.array(xg.correlation(model))

    # create a database to store time_axis, coerr_list, defect_list, avg_energy_list, eta_dot_field_list, eta_list, b_field_list
    data = {
        'Time': time_axis,
        'Defect': defect_list,
        'Avg_Energy': avg_energy_list,
        'eta_dot_field': eta_dot_field_list.tolist(), 
        'eta': eta_list.tolist(),
        'b_field': b_field_list.tolist(),
        'Distance': distances,
        'Coerr': coerr_list
    }
    xg.picture(model, "picture for beta = {beta} and b = {b}.png".format(beta=beta, b=b))
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    df.to_csv(f"results for beta = {beta} and b = {b}.csv", index=False)



if __name__ == "__main__":
    b_list = [2, 4, 6]
    beta_list = [0.9, 1.2]

    # Generate all combinations of b and beta
    combinations = list(itertools.product(b_list, beta_list))

    # Use multiprocessing to run simulations for all combinations of b and beta in parallel
    with multiprocessing.Pool(processes=len(combinations)) as pool:
        pool.map(run_simulation, combinations)

    print("All simulations completed.", flush=True)