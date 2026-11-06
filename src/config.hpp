#ifndef POISSON_CONFIG_HPP
#define POISSON_CONFIG_HPP
#include "types.hpp"
#include <iosfwd>
namespace poisson {
Config parse_arguments(int argc, char* argv[]);
void validate_config(const Config& config);
void print_usage(std::ostream& out);
bool help_requested(int argc, char* argv[]);
}
#endif
