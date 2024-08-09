# Demo fitting some real data on a graphene/hbn/PdBi2 tunnell barrier using the 'Maki model'
# Copyright (C) 2024 Lewis Powell
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with this program.
# If not, see https://www.gnu.org/licenses/. 

import sys
sys.path.append('..')
import fit_funcs_c as ff
import pandas as pd
import numpy as np

data = pd.read_csv("data/Sample_A_parallel_field.csv", index_col=0)
fields = data["B"].drop_duplicates().to_numpy()

experimental_temperature = 0.3 #Kelvin

gaps = np.zeros_like(fields)
zetas = np.zeros_like(fields)
pstart = (0.5, 0.01) # First parameters to try for fitting the first curve

for field_i, B in enumerate(fields):
    print(f'Fitting spectrum at {B:.2f} T')
    x, y = ff.get_spectrum(data, B)
    # Define a fitting function fixed to the experimental temperature
    model_maki = lambda x, D, Z : ff.maki_fit(x, # x data
                                              1, # normalisation
                                              D, # gap
                                              Z, # depairing parameter
                                              B, # field for zeeman splittng, fixed to experimental field
                                              experimental_temperature # temperature, fixed to experimental temperature
                                              )
    # After the first fit, use previous parameters as a first guess for the next fit
    if (field_i > 0):
        pstart = (gaps[field_i-1], zetas[field_i-1])
    # Do the actual fit and store results
    popt, pconv = ff.curve_fit(
        model_maki, # fitting model to use
        x, # x data
        y, # y data
        pstart, # starting parameters
        lower_bounds=np.array((0., 2e-5)),
        upper_bounds=np.array((1., 20)),
        loss='huber', # use huber loss to make fit more robust to outliers https://en.wikipedia.org/wiki/Huber_loss
        scale=0.1 # y scale for huber loss
        )
    print(f'Found optimal values:')
    print(f'Delta = {popt[0]:.3f} meV, zeta = {popt[1]:.3f}')
    gaps[field_i] = popt[0]
    zetas[field_i] = popt[1]

parameters = np.array([gaps, zetas])
parameter_headers = ['gap', 'zeta']
parameters_df = pd.DataFrame(data=parameters.transpose(), columns=parameter_headers, index=fields)
parameters_df.to_csv('output/Fit_parameters_SA_maki.csv')