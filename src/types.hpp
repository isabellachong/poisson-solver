#ifndef POISSON_TYPES_HPP
#define POISSON_TYPES_HPP
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(__CUDACC__)
#define POISSON_HD __host__ __device__
#else
#define POISSON_HD
#endif

namespace poisson {
using real = double;
using Int3 = std::array<int, 3>;
using FaceMask = std::array<bool, 6>;

struct Config {
    int dimension = 2;
    Int3 points{64, 64, 64};
    Int3 process_grid{0, 0, 0}; // 0 = choose automatically.
    int iterations = 500;
    int check_every = 0; // 0 = fixed iteration count.
    int warmup = 3;
    real rtol = 1e-8;
    real atol = 1e-10;
    bool overlap = false;
    int block_size = 256;
    std::string problem = "mixed";
    std::string communication = "cuda-aware";
    std::string solution_prefix;
};

// n = owned interior counts; offset = number of preceding global interior points.
// 2D uses k=0 and n[2]=1. 3D owned indices start at (1,1,1).
struct Grid {
    int dimension = 2;
    int n[3]{1, 1, 1};
    int offset[3]{0, 0, 0};
    int global[3]{1, 1, 1};
    POISSON_HD std::size_t row_stride() const { return static_cast<std::size_t>(n[0]) + 2; }
    POISSON_HD std::size_t plane_stride() const { return row_stride() * (static_cast<std::size_t>(n[1]) + 2); }
    POISSON_HD std::size_t index(int i, int j, int k = 0) const {
        return static_cast<std::size_t>(k) * plane_stride() + static_cast<std::size_t>(j) * row_stride() + i;
    }
    std::size_t size() const { return plane_stride() * (dimension == 3 ? static_cast<std::size_t>(n[2]) + 2 : 1); }
    std::size_t owned() const { return static_cast<std::size_t>(n[0]) * n[1] * n[2]; }
};

struct Region {
    int begin[3]{1, 1, 0};
    int end[3]{1, 1, 1}; // exclusive
    std::size_t size() const {
        if(end[0] <= begin[0] || end[1] <= begin[1] || end[2] <= begin[2]) return 0;
        return static_cast<std::size_t>(end[0]-begin[0]) * (end[1]-begin[1]) * (end[2]-begin[2]);
    }
};
struct Coefficients { real inverse_h2[3]{0, 0, 0}; real inverse_diagonal = 0; };
struct Fields { std::vector<real> solution, rhs; real rhs_inf = 0; };
struct Partition { int count; int offset; };
struct Device {
    int ordinal = -1;
    std::string name = "CPU";
    std::string uuid;
};
struct RunInfo {
    std::string mode = "serial";
    int rank = 0;
    int ranks = 1;
    Int3 process_grid{1,1,1};
    int gpus_used = 0;
    int max_ranks_per_gpu = 0;
    std::string device_name = "CPU";
};
struct Result {
    int iterations = 0;
    bool converged = false;
    real residual_inf = 0, rhs_inf = 0, threshold = 0, error_inf = 0, error_rms = 0;
    double setup_seconds = 0, solve_seconds = 0, diagnostics_seconds = 0;
};
}
#endif
