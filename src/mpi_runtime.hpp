#ifndef POISSON_MPI_RUNTIME_HPP
#define POISSON_MPI_RUNTIME_HPP
#include "runtime.hpp"
#include <mpi.h>
#include <array>
namespace poisson {
void mpi_check(int code, const char* operation);
class MpiRuntime final : public Runtime {
    MPI_Comm communicator_=MPI_COMM_NULL;
    Grid grid_;
    RunInfo info_;
    std::array<int,6> neighbours_{};
    std::array<int,6> counts_{};
    std::array<MPI_Request,12> requests_{};
    bool active_=false;
public:
    MpiRuntime(const Config& config, const Device& device, bool gpu);
    ~MpiRuntime();
    MpiRuntime(const MpiRuntime&)=delete;
    MpiRuntime& operator=(const MpiRuntime&)=delete;
    const Grid& grid() const { return grid_; }
    const RunInfo& info() const { return info_; }
    FaceMask active_faces() const;
    void begin_exchange(Backend& backend) override;
    void finish_exchange(Backend& backend) override;
    real maximum(real local) override;
    real sum(real local) override;
    void barrier() override;
    double now() const override { return MPI_Wtime(); }
    bool distributed_execution() const override { return true; }
};
}
#endif
