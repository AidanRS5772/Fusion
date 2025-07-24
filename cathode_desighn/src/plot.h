#ifndef PLOT_H
#define PLOT_H

#include <Eigen/Dense>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

using Vector6d = Eigen::Vector<double, 6>;
using Vector3d = Eigen::Vector3d;
using json = nlohmann::json;
using V_func = std::function<std::optional<double>(const Vector3d &)>;
using E_func = std::function<std::optional<Vector3d>(const Vector3d &)>;

inline void plot_mesh(const size_t hash) {
    std::string cmd = "node " + std::string(PROJECT_ROOT) + "/src/plots/plot.js 0 " + std::string(PROJECT_ROOT) + " " + std::to_string(hash);
    std::system(cmd.c_str());
}

inline void plot_mesh_path(const size_t hash, const std::vector<Vector6d> &path) {
    std::vector<double> X, Y, Z, S;
    X.reserve(path.size());
    Y.reserve(path.size());
    Z.reserve(path.size());
    S.reserve(path.size());
    for (const auto &p : path) {
        X.push_back(p[0]);
        Y.push_back(p[1]);
        Z.push_back(p[2]);
        S.push_back(p.tail<3>().norm());
    }

    json data = {{"X", X}, {"Y", Y}, {"Z", Z}, {"S", S}};
    std::ofstream plot_data(std::string(PROJECT_ROOT) + "/src/plots/plot_data.json");
    plot_data << data.dump();
    plot_data.close();

    std::string cmd = "node " + std::string(PROJECT_ROOT) + "/src/plots/plot.js 1 " + std::string(PROJECT_ROOT) + " " + std::to_string(hash);
    std::system(cmd.c_str());
}

inline void plot_mesh_electric_feild(const size_t hash, E_func E_feild, const double L, const size_t N = 20) {
    std::vector<double> X, Y, Z, U, V, W, M;
    X.reserve(N * N * N / 2);
    Y.reserve(N * N * N / 2);
    Z.reserve(N * N * N / 2);
    U.reserve(N * N * N / 2);
    V.reserve(N * N * N / 2);
    W.reserve(N * N * N / 2);
    M.reserve(N * N * N / 2);
    auto r = Eigen::VectorXd::LinSpaced(N, -L, L);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < N; k++) {
                Vector3d R(r[i], r[j], r[k]);
                if (R.norm() <= L) {
                    auto E = E_feild(R);
                    if (E.has_value()) {
                        Vector3d E_vec = E.value();
                        X.push_back(R.x());
                        Y.push_back(R.y());
                        Z.push_back(R.z());
                        U.push_back(E_vec.x());
                        V.push_back(E_vec.y());
                        W.push_back(E_vec.z());
                        M.push_back(E_vec.norm());
                    }
                }
            }
        }
    }

    json data = {{"X", X}, {"Y", Y}, {"Z", Z}, {"U", U}, {"V", V}, {"W", W}, {"M", M}};
    std::ofstream plot_data(std::string(PROJECT_ROOT) + "/src/plots/plot_data.json");
    plot_data << data.dump();
    plot_data.close();

    std::string cmd = "node " + std::string(PROJECT_ROOT) + "/src/plots/plot.js 2 " + std::string(PROJECT_ROOT) + " " + std::to_string(hash);
    std::system(cmd.c_str());
}

inline void plot_energy(std::vector<double> &times, std::vector<std::pair<double, double>> &energies) {
    std::vector<double> KE, PE, E;
    KE.reserve(energies.size());
    PE.reserve(energies.size());
    E.reserve(energies.size());
    for (const auto &energy : energies) {
        KE.push_back(energy.first);
        PE.push_back(energy.second);
        E.push_back(energy.first + energy.second);
    }

    json data = {{"KE", KE}, {"PE", PE}, {"E", E}, {"T", times}};
    std::ofstream plot_data(std::string(PROJECT_ROOT) + "/src/plots/plot_data.json");
    plot_data << data.dump();
    plot_data.close();

    std::string cmd = "node " + std::string(PROJECT_ROOT) + "/src/plots/plot.js 3 " + std::string(PROJECT_ROOT);
    std::system(cmd.c_str());
}

inline void plot_orbits(
    std::vector<double> &fusion_probs,
    std::function<double(double)> exp,
    std::function<double(double)> gamma,
    const size_t N = 500) {
    std::vector<double> exp_vals, gamma_vals, X;
    exp_vals.reserve(N);
    gamma_vals.reserve(N);
    X.reserve(N);
    auto eigenX = Eigen::VectorXd::LinSpaced(N, 0, *std::max_element(fusion_probs.begin(), fusion_probs.end()));
    for (const double x : eigenX) {
        exp_vals.push_back(exp(x));
        gamma_vals.push_back(gamma(x));
        X.push_back(x);
    }
    json data = {{"FUSION_PROBS", fusion_probs}, {"EXP", exp_vals}, {"GAMMA", gamma_vals}, {"X", X}};
    std::ofstream plot_data(std::string(PROJECT_ROOT) + "/src/plots/plot_data.json");
    plot_data << data.dump();
    plot_data.close();
    std::string cmd = "node " + std::string(PROJECT_ROOT) + "/src/plots/plot.js 4 " + std::string(PROJECT_ROOT);
    std::system(cmd.c_str());
}

#endif
