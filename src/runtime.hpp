#ifndef POISSON_RUNTIME_HPP
#define POISSON_RUNTIME_HPP
#include "backend.hpp"
#include <chrono>
namespace poisson {
class Runtime {
public:
    virtual ~Runtime()=default;
    virtual void begin_exchange(Backend& backend)=0;
    virtual void finish_exchange(Backend& backend)=0;
    virtual real maximum(real local)=0;
    virtual real sum(real local)=0;
    virtual void barrier()=0;
    virtual double now() const=0;
    virtual bool distributed_execution() const=0;
};
class SerialRuntime final : public Runtime {
public:
    void begin_exchange(Backend&) override {}
    void finish_exchange(Backend&) override {}
    real maximum(real local) override { return local; }
    real sum(real local) override { return local; }
    void barrier() override {}
    double now() const override {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    bool distributed_execution() const override { return false; }
};
}
#endif
