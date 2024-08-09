// Implement newton-raphson root finding for complex roots
// Copyright (C) 2024 Lewis Powell
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along with this program.
// If not, see https://www.gnu.org/licenses/. 
#pragma once
#include <vector>
#include <functional>
#include <complex>

template <typename function_type, typename derivative_type>
class newton_raphson_complex_solver
{
	const function_type function{}; // Function to evalutate
	const derivative_type derivative{}; // Derivative of function to evaluate;
public:
	newton_raphson_complex_solver(const function_type& function_init, const derivative_type& derivative_init) :
		function{ function_init },
		derivative{ derivative_init }
	{}

	std::complex<double> find_root(std::complex<double>& initial_value, const int max_evals, const double precision) {
		std::complex<double> result{initial_value};
		for (size_t iteration{}; iteration < max_evals; iteration++) {
			std::complex<double> func_value{ function(result) };
			result = result - func_value / derivative(result);
			if (std::abs(func_value) < precision)
				return result;
		}
		return result;
	}
};

// Provide procedural interface
template <typename function_type, typename derivative_type>
std::complex<double> solve_complex_newton_raphson(const function_type& function, const derivative_type& derivative, std::complex<double>& initial_value, const int max_evals = 100, const double precision = 0.00001 ) {
	return newton_raphson_complex_solver<function_type, derivative_type>(function, derivative).find_root(initial_value, max_evals, precision);
}