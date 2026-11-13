#ifndef POISSON_PROBLEM_HPP
#define POISSON_PROBLEM_HPP
#include "types.hpp"
namespace poisson {
Coefficients coefficients(const Grid& grid);
real exact_solution(const std::string& problem, int dimension, real x, real y, real z);
real forcing(const std::string& problem, int dimension, real x, real y, real z);
Fields initialize_fields(const Grid& grid, const std::string& problem);
Region owned_region(const Grid& grid);
Region interior_region(const Grid& grid);
std::vector<Region> boundary_regions(const Grid& grid);
std::size_t face_size(const Grid& grid, int face);
// Face = 2*axis + side, with side 0=minus, 1=plus.
POISSON_HD inline void face_point(const Grid& g, int face, std::size_t q, bool received,
                                  int& i, int& j, int& k) {
    const int axis = face / 2;
    const int fixed = received ? (face % 2 ? g.n[axis]+1 : 0) : (face % 2 ? g.n[axis] : 1);
    k = g.dimension == 3 ? 1 : 0;
    if(axis == 0) { i = fixed; j = static_cast<int>(q % g.n[1])+1; if(g.dimension == 3) k = static_cast<int>(q / g.n[1])+1; }
    else if(axis == 1) { j = fixed; i = static_cast<int>(q % g.n[0])+1; if(g.dimension == 3) k = static_cast<int>(q / g.n[0])+1; }
    else { k = fixed; i = static_cast<int>(q % g.n[0])+1; j = static_cast<int>(q / g.n[0])+1; }
}
// Find neighbours and calculate the stencil value at a point. 
POISSON_HD inline real stencil_value(const real* old_values, const real* rhs,
                                     std::size_t q, const Grid& g, const Coefficients& c) {
    real sum = rhs[q] + c.inverse_h2[0]*(old_values[q-1]+old_values[q+1])
                       + c.inverse_h2[1]*(old_values[q-g.row_stride()]+old_values[q+g.row_stride()]);
    if(g.dimension == 3) sum += c.inverse_h2[2]*(old_values[q-g.plane_stride()]+old_values[q+g.plane_stride()]);
    return sum * c.inverse_diagonal;
}
POISSON_HD inline real stencil_residual(const real* values, const real* rhs,
                                        std::size_t q, const Grid& g, const Coefficients& c) {
    real ahu = c.inverse_h2[0]*(2*values[q]-values[q-1]-values[q+1])
             + c.inverse_h2[1]*(2*values[q]-values[q-g.row_stride()]-values[q+g.row_stride()]);
    if(g.dimension == 3) ahu += c.inverse_h2[2]*(2*values[q]-values[q-g.plane_stride()]-values[q+g.plane_stride()]);
    return rhs[q] - ahu;
}
}
#endif
