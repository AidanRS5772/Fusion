#include "make_mesh.h"
#include "octo_tree.h"
#include "plot.h"
#include "solve_path.h"
#include "solve_pde.h"
#include <Eigen/Dense>
#include <chrono>
#include <gperftools/profiler.h>
#include <random>

constexpr int app_cnt = 362;
constexpr double cathode_radius = 5; // [cm]
constexpr double anode_radius = 25;  // [cm]
constexpr double wire_radius = .1;   // [cm]
constexpr double voltage = 1;        // [MV]
constexpr double mD = 2.08690083;    // [MeV][cm/ns]^-2 mass of a deuteron (m = E/c^2)
constexpr double temp = 28.5724675;  // [µeV][cm/ns]^-2 room temprature energy
constexpr size_t N = 10'000;

using namespace std;

vector<pair<Vector3d, Vector3d>> make_init_states(const double a, const double b, const size_t size) {
    mt19937 gen(314);
    uniform_real_distribution<double> u;
    normal_distribution<double> n(0, sqrt(temp / mD) * 1e-6);
    const double a3 = a * a * a;
    const double b3 = b * b * b;
    vector<pair<Vector3d, Vector3d>> init_states(size);
    for (size_t i = 0; i < size; ++i) {
        const double r = cbrt((b3 - a3) * u(gen) + a3);
        const double th = 2 * M_PI * u(gen);
        const double sphi = 1 - 2 * u(gen);
        const double cphi = sqrt(1 - sphi * sphi);
        const Vector3d pos(r * cphi * sin(th), r * cphi * cos(th), r * sphi);
        const Vector3d vel(n(gen), n(gen), n(gen));
        init_states[i] = {pos, vel};
    }
    return init_states;
}

int main() {
    auto mesh = MakeMesh(app_cnt, anode_radius, cathode_radius, wire_radius, 4, 24);
    auto pde_sol = SolvePDE(mesh.file_name, voltage, cathode_radius);

    Vector3d init_pos(0, 0, 17.5);
    Vector3d init_vel(0, 0, 0);
    auto init_states = make_init_states(anode_radius, cathode_radius, N);

    ProfilerStart("path_profile.prof");

    auto start = chrono::high_resolution_clock::now();
    auto path_sol = SolvePath(mesh.mesh_tree, pde_sol, mD, init_pos, init_vel, 1e4);
    // size_t cnt = 0;
    // for (auto &[pos, vel] : init_states) {
    //     auto path_sol = SolvePath(mesh.mesh_tree, pde_sol, mD, vel, pos, 1e4);
    //     cnt++;
    //     if (cnt % 100 == 0) {
    //         std::cout << "Progress: " << cnt << "/" << N << std::endl;
    //     }
    // }
    auto end = chrono::high_resolution_clock::now();

    ProfilerStop();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "duration: " << duration.count() << " ms" << endl;

    // plot_mesh_path(mesh.hash, path_sol.path_info.path);

    return 0;
}
