#include "make_mesh.h"
#include "plot.h"
#include "solve_path.h"
#include "solve_pde.h"

constexpr int app_cnt = 6;
constexpr double cathode_radius = 5; // [cm]
constexpr double anode_radius = 25;  // [cm]
constexpr double wire_radius = .2;   // [cm]
constexpr double voltage = 1;        // [MV]
constexpr double mD = 1875.612928;   // [MeV] mass of a deuteron (m = E/c^2)
constexpr double e = 898.7551787;    // [MeV/MV][cm/ns]^2 charge of an electron scalled by c^2

int main() {
    auto mesh = MakeMesh(app_cnt, anode_radius, cathode_radius, wire_radius, 4, 24);
    auto pde_sol = SolvePDE(mesh.file_name, voltage, cathode_radius, 3);

    auto V_func = [&pde_sol](const std::array<double, 3> &pos) { return pde_sol.V(pos); };
    auto E_func = [&pde_sol](const std::array<double, 3> &pos) { return pde_sol.E(pos); };
    auto path_sol = SolvePath(V_func, E_func, e, mD, {0, 0, 20}, {0, 0, 0});

    return 0;
}
