#include "make_mesh.h"
#include "octo_tree.h"
#include "plot.h"
#include "solve_path.h"
#include "solve_pde.h"
#include <Eigen/Dense>
#include <chrono>
#include <gperftools/profiler.h>
#include <random>

constexpr int app_cnt = 6;
constexpr double cathode_radius = 5; // [cm]
constexpr double anode_radius = 25;  // [cm]
constexpr double wire_radius = .2;   // [cm]
constexpr double voltage = 1;        // [MV]
constexpr double mD = 2.08690083;    // [MeV][cm/ns]^-2 mass of a deuteron (m = E/c^2)
std::mt19937 gen(314);
std::uniform_int_distribution<int> dis(1, 100);

std::vector<std::pair<Vector3d, Vector3d>> make_init_states(const size_t n) {
}

int main() {
    auto mesh = MakeMesh(app_cnt, anode_radius, cathode_radius, wire_radius, 4, 24);
    auto pde_sol = SolvePDE(mesh.file_name, voltage, cathode_radius);

    Vector3d init_pos(0, 0, 17.5);
    Vector3d init_vel(0, 0, 0);

    ProfilerStart("path_profile.prof");

    auto start = std::chrono::high_resolution_clock::now();
    auto path_sol = SolvePath(mesh.mesh_tree, pde_sol, mD, init_pos, init_vel, 1e4);
    auto end = std::chrono::high_resolution_clock::now();

    ProfilerStop();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Orbit Cnt: " << path_sol.orbit_cnt << std::endl;
    std::cout << "duration: " << duration.count() << " ms" << std::endl;

    // plot_mesh_path(mesh.hash, path_sol.path_info.path);

    return 0;
}
