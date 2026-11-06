#include "problem.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace poisson {
namespace { constexpr real pi = 3.141592653589793238462643383279502884; }
Coefficients coefficients(const Grid& g) {
    Coefficients c;
    real diagonal = 0;
    for(int axis=0; axis<g.dimension; ++axis) {
        const real scale = static_cast<real>(g.global[axis])+1;
        c.inverse_h2[axis] = scale*scale;
        diagonal += 2*c.inverse_h2[axis];
    }
    c.inverse_diagonal = 1/diagonal;
    return c;
}
real exact_solution(const std::string& problem, int dimension, real x, real y, real z) {
    if(problem == "zero") return 0;
    if(problem == "polynomial") {
        const real product = x*(1-x)*y*(1-y)*(dimension == 3 ? z*(1-z) : 1);
        return x+2*y+(dimension == 3 ? 3*z : 0)+product;
    }
    if(problem != "sine" && problem != "mixed") throw std::invalid_argument("Unknown problem");
    if(x == 0 || x == 1 || y == 0 || y == 1 || (dimension == 3 && (z == 0 || z == 1))) return 0;
    const real z_factor = dimension == 3 ? std::sin(pi*z) : 1;
    return (std::sin(pi*x)*std::sin(pi*y) + (problem == "mixed" ? 0.25*std::sin(2*pi*x)*std::sin(3*pi*y) : 0))*z_factor;
}
real forcing(const std::string& problem, int dimension, real x, real y, real z) {
    if(problem == "zero") return 0;
    if(problem == "polynomial") {
        const real a=x*(1-x), b=y*(1-y), c=z*(1-z);
        return dimension == 2 ? 2*(a+b) : 2*(a*b+a*c+b*c);
    }
    if(problem != "sine" && problem != "mixed") throw std::invalid_argument("Unknown problem");
    const real z_factor = dimension == 3 ? std::sin(pi*z) : 1;
    return pi*pi*(dimension*std::sin(pi*x)*std::sin(pi*y)
           +(problem == "mixed" ? 0.25*(dimension+11)*std::sin(2*pi*x)*std::sin(3*pi*y) : 0))*z_factor;
}
Fields initialize_fields(const Grid& g, const std::string& problem) {
    Fields f{std::vector<real>(g.size(),0), std::vector<real>(g.size(),0), 0};
    const int last_k=g.dimension == 3 ? g.n[2]+1 : 0;
    for(int k=0; k<=last_k; ++k)
        for(int j=0; j<=g.n[1]+1; ++j)
            for(int i=0; i<=g.n[0]+1; ++i) {
                const int index[3]{g.offset[0]+i,g.offset[1]+j,g.offset[2]+k};
                real coordinate[3]{0,0,0};
                bool physical=false;
                for(int a=0; a<g.dimension; ++a) {
                    coordinate[a]=static_cast<real>(index[a])/(static_cast<real>(g.global[a])+1);
                    physical=physical || index[a]==0 || index[a]==g.global[a]+1;
                }
                const auto q=g.index(i,j,k);
                if(physical) f.solution[q]=exact_solution(problem,g.dimension,coordinate[0],coordinate[1],coordinate[2]);
                if(i>=1 && i<=g.n[0] && j>=1 && j<=g.n[1] && (g.dimension==2 || (k>=1 && k<=g.n[2]))) {
                    f.rhs[q]=forcing(problem,g.dimension,coordinate[0],coordinate[1],coordinate[2]);
                    f.rhs_inf=std::max(f.rhs_inf,std::abs(f.rhs[q]));
                }
            }
    return f;
}
Region owned_region(const Grid& g) {
    return {{1,1,g.dimension==3 ? 1 : 0},{g.n[0]+1,g.n[1]+1,g.dimension==3 ? g.n[2]+1 : 1}};
}
Region interior_region(const Grid& g) {
    return {{2,2,g.dimension==3 ? 2 : 0},{g.n[0],g.n[1],g.dimension==3 ? g.n[2] : 1}};
}
std::vector<Region> boundary_regions(const Grid& g) {
    const auto all=owned_region(g);
    std::vector<Region> regions;
    // Assign intersections to the first axis that reaches them, avoiding duplicate updates.
    for(int axis=0; axis<g.dimension; ++axis) {
        Region low=all;
        for(int previous=0; previous<axis; ++previous) { low.begin[previous]=2; low.end[previous]=g.n[previous]; }
        low.end[axis]=2;
        if(low.size()) regions.push_back(low);
        if(g.n[axis]>1) {
            Region high=low;
            high.begin[axis]=g.n[axis]; high.end[axis]=g.n[axis]+1;
            if(high.size()) regions.push_back(high);
        }
    }
    return regions;
}
std::size_t face_size(const Grid& g, int face) {
    std::size_t count=1;
    for(int axis=0; axis<g.dimension; ++axis) if(axis!=face/2) count*=g.n[axis];
    return count;
}
}
