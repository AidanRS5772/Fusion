#include "fit_dist.h"
#include "make_mesh.h"
#include "octo_tree.h"
#include "plot.h"
#include "solve_path.h"
#include "solve_pde.h"
#include <Eigen/Dense>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <cmath>
#include <gperftools/profiler.h>
#include <ostream>
#include <random>

constexpr double cathode_radius = 5;      // [cm]
constexpr double anode_radius = 25;       // [cm]
constexpr double wire_radius = .1;        // [cm]
constexpr double voltage = 1;             // [MV]
constexpr double mD = 2.08690083;         // [MeV][cm/ns]^-2 mass of a deuteron (m = E/c^2)
constexpr double temp = 86.17333262;      // [µeV] room temprature energy
constexpr size_t mc_sample_size = 10'000; // number of monte carlo samples

constexpr double K_exp(const double r, const double R) {
    return voltage * (1 / r - 1.5 * (r + R) / (r * r + r * R + R * R)) / (1 / r - 1 / R) + 1.5 * temp * 1e-6;
}

double fusion_probability(const double K0) {
    constexpr double Ke = K_exp(cathode_radius, anode_radius);
    constexpr double Eg = 1875.612928 * M_PI * 0.0072973525643;
    constexpr std::array<double, 5> a = {54.6385, 270.405, -79.849, 15.41285, -1.166645};

    boost::math::quadrature::exp_sinh<double> integrator;
    const double f_int = integrator.integrate([K0, a](const double E) {
        const double S = a[0] + E * (a[1] + E * (a[2] + E * (a[3] + a[4] * E)));
        return std::sqrt(E) * S * std::sinh(3 * std::sqrt(2 * K0 * E) / Ke) * std::exp(-3 * E / Ke - std::sqrt(Eg / E));
    });

    const double den = std::sqrt(K0) +
                       std::sqrt(3 * M_PI / (2 * Ke)) * (K0 + Ke / 3) * std::erf(std::sqrt(3 * K0 / (2 * Ke))) * std::exp(3 * K0 / (2 * Ke));
    return f_int / den;
}

template <size_t N, size_t M>
std::pair<Eigen::Matrix<double, N + 1, 1>, double> poly_reg(const std::function<double(double)> f, const double a, const double b) {
    Eigen::Matrix<double, M, 1> Y;
    Eigen::Matrix<double, M, N + 1> V;
    std::vector<double> X(M);
    for (size_t i = 0; i < M; i++) {
        const double x = a + i * (b - a) / (M - 1);
        Y(i) = f(x);
        for (size_t j = 0; j < N + 1; j++)
            V(i, j) = std::pow(x, j);
    }

    Eigen::Matrix<double, N + 1, 1> coefs = (V.transpose() * V).ldlt().solve(V.transpose() * Y);
    double res = std::sqrt((V * coefs - Y).squaredNorm() / M);
    return {coefs, res};
}

double exp_max_energy() {
    constexpr double gamma = 0.577215664901532;
    return voltage + 1e-6 * temp * (std::log(2) - std::log(M_PI) / 2 + gamma + std::log(mc_sample_size) + std::log(std::log(mc_sample_size)));
}

std::vector<std::pair<Vector3d, Vector3d>> make_init_states() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> u;
    std::normal_distribution<double> n(0, std::sqrt(temp / mD) * 1e-6);
    constexpr double r3 = cathode_radius * cathode_radius * cathode_radius;
    constexpr double R3 = anode_radius * anode_radius * anode_radius;
    std::vector<std::pair<Vector3d, Vector3d>> init_states(mc_sample_size);
    for (size_t i = 0; i < mc_sample_size; ++i) {
        const double r = cbrt((r3 - R3) * u(gen) + r3);
        const double th = 2 * M_PI * u(gen);
        const double sphi = 1 - 2 * u(gen);
        const double cphi = sqrt(1 - sphi * sphi);
        const Vector3d pos(r * cphi * sin(th), r * cphi * cos(th), r * sphi);
        const Vector3d vel(n(gen), n(gen), n(gen));
        init_states[i] = {pos, vel};
    }
    return init_states;
}

void collect_data(const size_t app_cnt, const std::function<double(double)> fprob) {
    auto mesh = MakeMesh(app_cnt, anode_radius, cathode_radius, wire_radius, 4, 24);
    auto pde_sol = SolvePDE(mesh.file_name, voltage, cathode_radius);
    auto init_states = make_init_states();

    std::vector<double> fusion_probs;
    fusion_probs.reserve(mc_sample_size);
    std::vector<std::pair<Vector3d, Vector3d>> eternal_states;
    size_t zero_orbit_cnt = 0;

    size_t completed = 0;
    const size_t update_interval = std::max(1UL, mc_sample_size / 100);
    for (auto const &[pos, vel] : init_states) {
        auto path_sol = SolvePath(mesh.mesh_tree, pde_sol, mD, pos, vel, 1e4);
        if (path_sol.sucsessful) {
            if (path_sol.eternal) {
                eternal_states.push_back({pos, vel});
            } else {
                if (path_sol.orbit_cnt > 0) {
                    const double init_energy = path_sol.init_KE + voltage + path_sol.init_PE;
                    const double fusion_prob = path_sol.orbit_cnt * fprob(init_energy);
                    fusion_probs.push_back(fusion_prob);
                } else {
                    zero_orbit_cnt++;
                }
            }
        }

        completed++;
        if (completed % update_interval == 0) {
            double percent = 100.0 * completed / mc_sample_size;
            std::cout << "\rProgress: " << completed << "/" << mc_sample_size << " (" << std::fixed << std::setprecision(1) << percent << "%)"
                      << std::flush;
        }
    }

    ExpDist exp_fit(fusion_probs);
    GammaDist gamma_fit(fusion_probs);

    std::cout << std::setprecision(6) << std::endl;
    std::cout << "\nExponential Distribution: " << std::endl;
    std::cout << "\tlambda: " << exp_fit.lambda << std::endl;
    std::cout << "\tKS: " << exp_fit.ks << std::endl;

    std::cout << "\nGamma Distribution: " << std::endl;
    std::cout << "\talpha: " << gamma_fit.alpha << std::endl;
    std::cout << "\tlambda: " << gamma_fit.lambda << std::endl;
    std::cout << "\tKS: " << gamma_fit.ks << std::endl;

    std::cout << std::setprecision(2) << std::endl;
    std::cout << "\nZero Orbit Proportion: " << 100 * static_cast<double>(zero_orbit_cnt) / mc_sample_size << "% (" << zero_orbit_cnt << "/"
              << mc_sample_size << ")" << std::endl;
    std::cout << "Eternal States: " << 100 * static_cast<double>(eternal_states.size()) / mc_sample_size << "% (" << eternal_states.size() << "/"
              << mc_sample_size << ")" << std::endl;

    std::cout << std::setprecision(12) << std::endl;
    for (auto const &[pos, vel] : eternal_states) {
        std::cout << "\t[ " << pos.transpose() << " ] , [ " << vel.transpose() << " ]" << std::endl;
    }

    plot_orbits(
        fusion_probs,
        [exp_fit](double x) {
            return exp_fit.pdf(x);
        },
        [gamma_fit](double x) {
            return gamma_fit.pdf(x);
        });
}

int main() {
    constexpr size_t fprob_sample = 1'000;
    const double max_energy = exp_max_energy();
    auto [coefs, res] = poly_reg<2, fprob_sample>(fusion_probability, 1e-12, max_energy);
    std::cout << "Coeficents: " << coefs.transpose() << std::endl;
    auto fprob = [coefs](const double E) {
        return coefs[0] + E * (coefs[1] + coefs[2] * E);
    };
    collect_data(6, fprob);

    return 0;
}
