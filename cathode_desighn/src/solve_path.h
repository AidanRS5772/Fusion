#ifndef SOLVE_PATH_H
#define SOLVE_PATH_H

#include "solve_pde.h"
#include <Eigen/Dense>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/stepper/symplectic_euler.hpp>
#include <optional>
#include <stdexcept>

namespace odeint = boost::numeric::odeint;
using Vector3d = Eigen::Vector3d;
class SolvePath {
  private:
    class Hamiltonian {
      private:
        const SolvePDE &PDE_SOL;
        const double mass;

      public:
        Hamiltonian(const SolvePDE &PDE_SOL, double mass) : PDE_SOL(PDE_SOL), mass(mass) {}

        double H(const Vector3d &q, const Vector3d &p) const {
            auto opt_V = PDE_SOL.V(q);
            if (!opt_V.has_value()) {
                throw std::runtime_error("Out of Bounds Eval");
            }
            return opt_V.value() + p.squaredNorm() / (2 * mass);
        }

        void operator()(const Vector3d &q, Vector3d &dq, const Vector3d &p, double t) {
            dq = p / mass;
        }

        void operator()(const Vector3d &q, const Vector3d &p, Vector3d &dp, double t) {
            auto E_opt = PDE_SOL.E(q);
            if (!E_opt.has_value()) {
                throw std::runtime_error("Out of Bounds Eval");
            }
            dp = E_opt.value();
        }
    };

    struct PathInfo {
        bool active = false;
        std::vector<std::pair<Vector3d, Vector3d>> states{};
        std::vector<double> times{};
        std::vector<double> KE{};
        std::vector<double> PE{};

        PathInfo() = default;

        PathInfo(const size_t reserve_size) {
            states.reserve(reserve_size);
            KE.reserve(reserve_size);
            PE.reserve(reserve_size);
            active = true;
        }

        void add(const std::optional<std::pair<Vector3d, Vector3d>> state,
                 const std::optional<double> ke,
                 const std::optional<double> pe,
                 const std::optional<double> time) {
            if (state.has_value()) {
                states.push_back(state.value());
            } else {
                states.push_back(states.back());
            }
            if (time.has_value()) {
                times.push_back(time.value());
            } else {
                times.push_back(times.back());
            }
            if (ke.has_value() && pe.has_value()) {
                KE.push_back(ke.value());
                PE.push_back(pe.value());
            } else if (ke.has_value() && !pe.has_value()) {
                KE.push_back(ke.value());
                PE.push_back(KE.back() + PE.back() - ke.value());
            } else if (!ke.has_value() && pe.has_value()) {
                KE.push_back(KE.back() + PE.back() - ke.value());
                PE.push_back(pe.value());
            } else {
                KE.push_back(KE.back());
                PE.push_back(PE.back());
            }
        }
    };

    const SolvePDE &PDE_SOL;
    const double mass;
    const bool record_path;
    const double max_dt;
    const double max_time;

    std::optional<double> prev_prev_norm;
    std::optional<double> prev_norm;

    double H(const Vector3d &q, const Vector3d &p) const {
        auto opt_V = PDE_SOL.V(q);
        if (!opt_V.has_value()) {
            throw std::runtime_error("Out of Bounds Eval");
        }
        return opt_V.value() + p.squaredNorm() / (2 * mass);
    }

    void find_path(const Vector3d &init_q, const Vector3d &init_p) {
        constexpr int MAX_RETRY = 5;
        constexpr double STEP_REDUCTION = 0.5;
        constexpr double STEP_INCREASE = 1.5;
        constexpr double ENERGY_REL_TOL = 0.01;

        double t = 0.0;
        double dt = max_dt;
        auto state = std::make_pair(init_q, init_p);
        odeint::symplectic_euler<Vector3d> stepper;
        while (t <= max_time) {
            const auto cur_state = state;
            for (int retry_cnt = 0; retry_cnt <= MAX_RETRY; retry_cnt++) {
                try {
                    stepper.do_step(system, state, t, dt);
                    if (std::abs(H(state.first, state.second) / init_energy - 1) < ENERGY_REL_TOL) {
                        retry_cnt--;
                        state = cur_state;
                        dt *= STEP_REDUCTION;
                        continue;
                    } else {
                        dt = std::min(dt * STEP_INCREASE, max_dt);
                        break;
                    }
                } catch (std::exception &e) {
                    if (std::string(e.what()) == "Out of Bounds Eval") {
                        if (retry_cnt == MAX_RETRY) {
                            state = cur_state;
                            dt *= STEP_REDUCTION;
                        } else {
                            sucsessful = true;
                            return;
                        }
                    } else {
                        return;
                    }
                }
            }
            t += dt;

            if (record_path) {
                path_info.add(state, state.second.squaredNorm() / (2 * mass), std::nullopt, t);
            }

            double cur_norm = state.first.norm();
            if (prev_norm.has_value() && prev_prev_norm.has_value()) {
                if (prev_norm.value() < cur_norm && prev_prev_norm.value() < cur_norm) {
                    orbit_cnt++;
                }
            }
            prev_prev_norm = prev_norm;
            prev_norm = cur_norm;
        }

        sucsessful = true;
        eternal = false;
    }

  public:
    double init_energy;
    double init_KE;
    double init_PE;
    PathInfo path_info;
    bool sucsessful;
    bool eternal;
    size_t orbit_cnt;

    SolvePath(const SolvePDE &PDE_SOL,
              const double mass,
              const Vector3d init_pos,
              const Vector3d init_mom,
              const double max_dt = 0.1,
              const double max_time = 1e4,
              const bool record_path = false)
        : system(PDE_SOL, mass), mass(mass), record_path(record_path), max_dt(max_dt),
          max_time(max_time), sucsessful(false), eternal(false) {
        if (PDE_SOL.init_cache(init_pos)) {
            init_KE = init_mom.squaredNorm() / (2 * mass);
            init_PE = PDE_SOL.V(init_pos).value();
            init_energy = init_KE + init_PE;
            std::pair<Vector3d, Vector3d> init_state;
            if (record_path) {
                path_info = PathInfo(1000);
                path_info.add(init_state, init_KE, init_PE, 0);
            }
        }
    }
};

#endif
