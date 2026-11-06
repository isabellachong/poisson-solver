#include "config.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <ostream>
#include <stdexcept>
namespace poisson {
namespace {
std::string trim(const std::string& text) {
    const auto first=text.find_first_not_of(" \t\r\n");
    return first==std::string::npos ? "" : text.substr(first,text.find_last_not_of(" \t\r\n")-first+1);
}
int integer(const std::string& text) {
    std::size_t end=0; const auto value=std::stoi(text,&end);
    if(end!=text.size()) throw std::invalid_argument("Expected an integer: "+text);
    return value;
}
real number(const std::string& text) {
    std::size_t end=0; const auto value=std::stod(text,&end);
    if(end!=text.size()) throw std::invalid_argument("Expected a number: "+text);
    return value;
}
void set(Config& c, std::string key, const std::string& value) {
    std::replace(key.begin(),key.end(),'-','_');
    if(key=="dim" || key=="dimension") c.dimension=integer(value);
    else if(key=="nx") c.points[0]=integer(value);
    else if(key=="ny") c.points[1]=integer(value);
    else if(key=="nz") c.points[2]=integer(value);
    else if(key=="px") c.process_grid[0]=integer(value);
    else if(key=="py") c.process_grid[1]=integer(value);
    else if(key=="pz") c.process_grid[2]=integer(value);
    else if(key=="iterations") c.iterations=integer(value);
    else if(key=="check_every") c.check_every=integer(value);
    else if(key=="warmup") c.warmup=integer(value);
    else if(key=="rtol") c.rtol=number(value);
    else if(key=="atol") c.atol=number(value);
    else if(key=="block_size") c.block_size=integer(value);
    else if(key=="overlap") {
        if(value!="0" && value!="1") throw std::invalid_argument("overlap must be 0 or 1");
        c.overlap=value=="1";
    }
    else if(key=="problem") c.problem=value;
    else if(key=="communication") c.communication=value;
    else if(key=="solution_prefix") c.solution_prefix=value;
    else throw std::invalid_argument("Unknown option: "+key);
}
void read_file(const std::string& path, Config& c) {
    std::ifstream file(path);
    if(!file) throw std::runtime_error("Cannot open configuration file: "+path);
    std::string line; int line_number=0;
    while(std::getline(file,line)) {
        ++line_number; line=trim(line.substr(0,line.find('#')));
        if(line.empty()) continue;
        if(line=="end") break;
        const auto equals=line.find('=');
        if(equals==std::string::npos) throw std::invalid_argument(path+":"+std::to_string(line_number)+": expected key=value");
        try { set(c,trim(line.substr(0,equals)),trim(line.substr(equals+1))); }
        catch(const std::exception& e) { throw std::invalid_argument(path+":"+std::to_string(line_number)+": "+e.what()); }
    }
    if(file.bad()) throw std::runtime_error("Error reading configuration file: "+path);
}
}
bool help_requested(int argc, char* argv[]) {
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--help" || std::string(argv[i])=="-h") return true;
    return false;
}
Config parse_arguments(int argc, char* argv[]) {
    Config c;
    // Read explicitly requested configuration files first, then apply CLI overrides.
    for(int i=1;i<argc;++i) if(std::string(argv[i])=="--config") {
        if(++i>=argc) throw std::invalid_argument("--config requires a filename");
        read_file(argv[i],c);
    }
    for(int i=1;i<argc;++i) {
        const std::string flag=argv[i];
        if(flag=="--config") { ++i; continue; }
        if(flag.size()<3 || flag.substr(0,2)!="--") throw std::invalid_argument("Expected --option value: "+flag);
        if(++i>=argc) throw std::invalid_argument("Missing value for "+flag);
        set(c,flag.substr(2),argv[i]);
    }
    validate_config(c); return c;
}
void validate_config(const Config& c) {
    if(c.dimension!=2 && c.dimension!=3) throw std::invalid_argument("--dim must be 2 or 3");
    std::size_t allocation=1;
    std::uint64_t total=1;
    for(int a=0;a<3;++a) {
        if(c.process_grid[a]<0) throw std::invalid_argument("Process-grid dimensions must be nonnegative");
        if(a>=c.dimension) continue;
        if(c.points[a]<1 || c.points[a]>std::numeric_limits<int>::max()-2) throw std::invalid_argument("Grid dimensions must be positive and below INT_MAX-1");
        const auto extent=static_cast<std::size_t>(c.points[a])+2;
        if(allocation>std::numeric_limits<std::size_t>::max()/sizeof(real)/extent)
            throw std::invalid_argument("Grid size exceeds addressable memory");
        allocation*=extent;
        if(total>std::numeric_limits<std::uint64_t>::max()/7/static_cast<unsigned>(c.points[a]))
            throw std::invalid_argument("Grid count exceeds supported reporting range");
        total*=static_cast<unsigned>(c.points[a]);
    }
    if(c.dimension==2 && c.process_grid[2]>1) throw std::invalid_argument("2D problems require pz=1 or automatic (0)");
    if(c.iterations<0 || c.warmup<0 || c.check_every<0) throw std::invalid_argument("Iteration counts must be nonnegative");
    if(!std::isfinite(c.rtol) || !std::isfinite(c.atol) || c.rtol<0 || c.atol<0) throw std::invalid_argument("Tolerances must be finite and nonnegative");
    if(c.block_size<32 || c.block_size>1024 || c.block_size%32) throw std::invalid_argument("block-size must be a multiple of 32 from 32 through 1024");
    if(c.problem!="sine" && c.problem!="mixed" && c.problem!="polynomial" && c.problem!="zero") throw std::invalid_argument("Unknown manufactured problem");
    if(c.communication!="cuda-aware" && c.communication!="host") throw std::invalid_argument("communication must be cuda-aware or host");
}
void print_usage(std::ostream& out) {
    out << "Jacobi-Poisson 2D/3D: poisson_serial and poisson_mpi (CUDA enabled by default)\n"
        << "Options (defaults in parentheses):\n"
        << "  --config FILE                    Optional key=value settings\n"
        << "  --dim 2|3 --nx N --ny N --nz N   Interior grid counts (2; 64 each)\n"
        << "  --px X --py Y --pz Z            MPI process grid (0=automatic)\n"
        << "  --iterations K --check-every Q   Fixed/max iterations (500); 0=fixed (0)\n"
        << "  --rtol R --atol A                Residual tolerances (1e-8; 1e-10)\n"
        << "  --warmup K --overlap 0|1         Untimed iterations (3); overlap (0)\n"
        << "  --problem sine|mixed|polynomial|zero (mixed)\n"
        << "  --communication cuda-aware|host GPU boundary transfer method (cuda-aware)\n"
        << "  --block-size N                   CUDA threads per block (256)\n"
        << "  --solution-prefix PATH           Optional per-rank CSV output (empty)\n"
        << "Distributed runs must split at least two axes; one-rank baselines are allowed.\n"
        << "CUDA devices are detected automatically; local ranks are assigned cyclically.\n";
}
}
