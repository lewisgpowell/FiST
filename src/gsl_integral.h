// Wrapper around the gsl integral function
// Copyright (C) 2024 Lewis Powell
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along with this program.
// If not, see https://www.gnu.org/licenses/. 
#pragma once
#include <memory>
#include <cmath>
#include <utility>
#include <functional>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

template <typename function_type>
class gsl_integral
{
	const function_type function{}; // Function to evaluate
	const int workspace_size{};
	// Pointer to the workspace, use gsl_integration_workspace_free as deleter
	std::unique_ptr < gsl_integration_workspace, std::function < void(gsl_integration_workspace*) >> workspace;

	// Wrapper which uses the required function template for gsl integration but casts void pointer to this class and evaluates function
	static double gsl_wrapper(double x, void* p)
	{
        gsl_integral* t{ static_cast<gsl_integral*>(p) };
		return t->function(x);
	}

public:
	gsl_integral(const function_type& function_init, const int workspace_size_init):
		function{function_init},
		workspace_size{workspace_size_init},
		workspace{gsl_integration_workspace_alloc(workspace_size_init), gsl_integration_workspace_free}
	{}

	double integrate(const double min, const double max, const double epsabs, const double epsrel)
	{
		gsl_function gsl_func{};
		gsl_func.function = &gsl_wrapper;
		gsl_func.params = this;
        double result{}; double error{};
        gsl_set_error_handler_off();
        // Perform integration using routine depending on bounds
        if (!std::isinf(min) && !std::isinf(max))
        {
            gsl_integration_qags(&gsl_func, min, max,
                epsabs, epsrel, workspace_size,
                workspace.get(), &result, &error);
        }
        else if (std::isinf(min) && !std::isinf(max))
        {
            gsl_integration_qagil(&gsl_func, max,
                epsabs, epsrel, workspace_size,
                workspace.get(), &result, &error);
        }
        else if (!std::isinf(min) && std::isinf(max))
        {
            gsl_integration_qagiu(&gsl_func, min,
                epsabs, epsrel, workspace_size,
                workspace.get(), &result, &error);
        }
        else
        {
            gsl_integration_qagi(&gsl_func,
                epsabs, epsrel, workspace_size,
                workspace.get(), &result, &error);
        }
        return result;
	}
};

// Provide procedural interface
template <typename function_type>
double integrate(const function_type& function, const std::pair<double, double>& range, const double epsabs = 0, const double epsrel = 1e-4, int limit = 10000)
{
    return gsl_integral<function_type>(function, limit).integrate(range.first, range.second, epsabs, epsrel);
}

