
# SIMULATION

# Lunar Leaper: Gravimetry Modelling
# Yara Luginbühl

import os
import sys
import matplotlib.pyplot as plt
import pygimli as pg
import time
import numpy as np
from itertools import product
import pandas as pd

sys.path.append("C:/Users/yslug/Desktop/SpaceSystems/LunarLeaper/ModellingCode")
from ModelFunction import LavaTubeGravimetry


# Reference Model:
x_ref, gfinal_ref, gcorr_ref, model_ref = LavaTubeGravimetry(tube_radius = 200, tube_depth = 110, regolith_thickness = 0)

df = pd.DataFrame({'Position x': x_ref})
df['gcorr_Reference [mGal]'] = gfinal_ref

# Simulation parameters


sim_nr = 2
date = '2025-07-06'
rad_min = 50
rad_max = 300
rad_step = 5

dep_min = 25
dep_max = 55
dep_step = 1


radii = np.arange(rad_min, rad_max, rad_step)
depths = np.arange(dep_min, dep_max, dep_step)



misfit_matrix = np.zeros((len(radii), len(depths)))


# Simulation
import time
start = time.time()
for i, radius in enumerate(radii):
    for j, depth in enumerate(depths):
        print("Run", radius, depth)
        # Replace this with your actual model function
        startrun = time.time()
        _,_,gcorr_model,_  = LavaTubeGravimetry(radius, depth, 0)
        endrun = time.time()

        # noise_inst = np.random.randn(gcorr_model.shape[0]) * 0.05
        # gcorr_model_noisy = gcorr_model + noise_inst

        df[f'gcorr_R{radius}_D{depth}'] = gcorr_model

        # Store the result
        misfit_matrix[i, j] = np.sum((gcorr_model - gcorr_ref) ** 2) /len(gcorr_ref)
        print("Runtime Run", endrun - startrun)

end = time.time()

print("Time for simulation:",end-start)

plt.imshow(misfit_matrix, origin='lower', aspect='auto', extent=[depths[0], depths[-1], radii[0], radii[-1]])
plt.xlabel('Depth')
plt.ylabel('Radius')
plt.colorbar(label='Misfit')
plt.title('Misfit Matrix')
plt.savefig(f"C:/Users/yslug/Desktop/SpaceSystems/LunarLeaper/ModellingCode/Simulation/Sim{sim_nr}/Misfit_Simulation{sim_nr}_Runtime{end-start:.2f}s.png")



min_indices = np.where(misfit_matrix == np.min(misfit_matrix))

row_indices = min_indices[0]
col_indices = min_indices[1]

for i in range(len(row_indices)):
    r = radii[row_indices[i]]
    d = depths[col_indices[i]]
    print(f"Minimum misfit at radius={r}, depth={d}")

with open(f"C:/Users/yslug/Desktop/SpaceSystems/LunarLeaper/ModellingCode/Simulation/Sim{sim_nr}/Simulation{sim_nr}_Results.csv", "w") as f:
    f.write(f"Simulation {sim_nr} Results\n")
    f.write("Run Date: {date} \n")
    f.write(f"Parameters: R from {rad_min} to {rad_max} with stepsize {rad_step}, d from {dep_min} to {dep_max} with stepsize {dep_step}\n")
    f.write(f"Reference: R = 200, d = 110 \n")
    f.write("\n")  # Blank line before table

    df.to_csv(f, index=False)

