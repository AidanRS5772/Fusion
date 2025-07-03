#include "make_mesh.h"
#include "solve_path.h"
#include "solve_pde.h"
#include <Eigen/Dense>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <chrono>
#include <cmath>
#include <gperftools/profiler.h>
#include <iostream>
#include <random>

using namespace std;

constexpr int app_cnt = 6;
constexpr double cathode_radius = 5;  // [cm]
constexpr double anode_radius = 25;   // [cm]
constexpr double wire_radius = .2;    // [cm]
constexpr double voltage = 1;         // [MV]
constexpr double temprature = 295;    // [K]
constexpr double mD = 2.08690083;     // [MeV][cm/ns]^-2 mass of a deuteron (m = E/c^2)
constexpr double kB = 8.61733326e-11; // [MeV/K][cm/ns]^-2 boltzmans constant;
constexpr size_t chunk = 10;
mt19937 gen(314);
std::uniform_real_distribution<double> dist(0, 1);

template <size_t N> std::pair<std::array<Eigen::Vector3d, N>, std::array<Eigen::Vector3d, N>> make_samples() {
    std::array<Eigen::Vector3d, N> pos, vel;
    auto vel_dist = [](double u) {
        return sqrt((2 * kB * temprature) / mD) * boost::math::erf_inv(2 * u - 1);
    };
    const double r_cubed = cathode_radius * cathode_radius * cathode_radius;
    const double R_cubed = anode_radius * anode_radius * anode_radius;
    for (size_t i = 0; i < N; i++) {
        const double r = cbrt((R_cubed - r_cubed) * dist(gen) + r_cubed);
        const double cphi = 1 - 2 * dist(gen);
        const double sphi = std::sqrt(1 - cphi * cphi);
        const double theta = 2 * M_PI * dist(gen);
        pos[i] = Eigen::Vector3d(r * sphi * cos(theta), r * sphi * sin(theta), r * cphi);
        vel[i] = Eigen::Vector3d(vel_dist(dist(gen)), vel_dist(dist(gen)), vel_dist(dist(gen)));
    }

    return {pos, vel};
}

int main() {
    auto mesh = MakeMesh(app_cnt, anode_radius, cathode_radius, wire_radius, 4, 24);
    auto pde_sol = SolvePDE<chunk>(mesh.file_name, voltage, cathode_radius);

    const auto [init_pos, init_vel] = make_samples<chunk>();

    ProfilerStart("path_profile.prof");

    auto start = chrono::high_resolution_clock::now();
    auto path_sol = SolveMultiPath<chunk>(*mesh.mesh_tree, pde_sol, mD, init_pos, init_vel);
    auto end = chrono::high_resolution_clock::now();

    ProfilerStop();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "duration: " << duration.count() << " ms" << endl;
    return 0;
}
