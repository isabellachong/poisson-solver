#include "decomposition.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace poisson {
Partition partition_axis(int total, int parts, int coordinate) {
    if(total<1 || parts<1 || parts>total || coordinate<0 || coordinate>=parts)
        throw std::invalid_argument("Invalid partition: each rank must own at least one point per active direction");
    return {total/parts+(coordinate<total%parts), coordinate*(total/parts)+std::min(coordinate,total%parts)};
}
Int3 choose_process_grid(const Config& c, int ranks) {
    Int3 best{0,0,0};
    double best_score=std::numeric_limits<double>::infinity();
    for(int px=1; px<=ranks && px<=c.points[0]; ++px) {
        if(ranks%px || (c.process_grid[0] && c.process_grid[0]!=px)) continue;
        const int remaining=ranks/px;
        for(int py=1; py<=remaining && py<=c.points[1]; ++py) {
            if(remaining%py || (c.process_grid[1] && c.process_grid[1]!=py)) continue;
            const int pz=remaining/py;
            if(c.dimension==2 && pz!=1) continue;
            if(c.dimension==3 && pz>c.points[2]) continue;
            if(c.process_grid[2] && c.process_grid[2]!=pz) continue;
            if(ranks>1 && (int(px>1)+int(py>1)+int(pz>1))<2) continue;
            double score=static_cast<double>(px)/c.points[0]+static_cast<double>(py)/c.points[1];
            if(c.dimension==3) score+=static_cast<double>(pz)/c.points[2];
            if(score<best_score-1e-14) { best={px,py,pz}; best_score=score; }
        }
    }
    if(best[0]==0) throw std::invalid_argument("No valid multidimensional process grid. Use one rank, or split at least two axes (e.g. 2x2 or 2x2x2); check --px/--py/--pz and grid sizes.");
    return best;
}
Grid make_grid(const Config& c, Int3 dimensions, Int3 coordinates) {
    Grid g; g.dimension=c.dimension;
    for(int a=0; a<c.dimension; ++a) {
        const auto p=partition_axis(c.points[a],dimensions[a],coordinates[a]);
        g.n[a]=p.count; g.offset[a]=p.offset; g.global[a]=c.points[a];
    }
    return g;
}
std::uint64_t unknown_count(const Config& c) {
    std::uint64_t count=1;
    for(int a=0; a<c.dimension; ++a) count*=static_cast<std::uint64_t>(c.points[a]);
    return count;
}
std::uint64_t nonzero_count(const Config& c) {
    const auto total=unknown_count(c);
    std::uint64_t count=total;
    for(int a=0; a<c.dimension; ++a) count+=2*(total-total/static_cast<std::uint64_t>(c.points[a]));
    return count;
}
}
