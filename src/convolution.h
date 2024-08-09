// General convolution of density of states with the derivative of the Fermi-Dirac distribution
// Copyright (C) 2024 Lewis Powell
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along with this program.
// If not, see https://www.gnu.org/licenses/. 

#pragma once
#include <functional>
#include <memory>
#include "gsl_integral.h"
#include "Fitting_Funcs.h"

// Overwrites input data with output
template <class parameter_container, class function_type>
void convolution(const function_type& density_of_states, double* data, size_t num_points, const double T,
	const parameter_container& density_of_states_params) {
	// Iterate over points to evaluate convolution at
	for (size_t point_i{}; point_i < num_points; point_i++) {
		// Define the funtion to integrate using all the parameters
		const double V{ data[point_i] };
		data[point_i] = integrate(
			[&density_of_states, &V, &T, &density_of_states_params](const double E) {
				return -fermi_dirac_deriv(E + V, T) * density_of_states(E, density_of_states_params);
			},
			{ -10, 10 }
		);
	}
}
