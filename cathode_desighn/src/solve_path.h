#ifndef SOLVE_PATH_H
#define SOLVE_PATH_H

#include "octo_tree.h"
#include "solve_pde.h"
#include <Eigen/Dense>
#include <boost/numeric/odeint.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>

namespace odeint = boost::numeric::odeint;
using Vector3d = Eigen::Vector3d;
using Vector6d = Eigen::Vector<double, 6>;

class SolvePath {
    struct Path {
        std::vector<Vector6d> path{};
        std::vector<double> times{};
        std::vector<std::pair<double, double>> energies{};
    };

  public:
    Path path_info;
    size_t orbit_cnt = 0;
    bool eternal = false;
    bool sucsessful = false;
    double init_energy;
    double init_KE;
    double init_PE;

    SolvePath(
        const std::unique_ptr<OctoTree> &mesh_tree_,
        const SolvePDE &PDE_,
        double mass_,
        const Vector3d &init_pos,
        const Vector3d &init_vel,
        const double max_t = 1e3,
        const bool record_path_ = false)
        : mesh_tree(*mesh_tree_), PDE(PDE_), mass(mass_), record_path(record_path_) {
        if (PDE.init_cache(init_pos)) {
            init_KE = (0.5 * mass) * init_vel.dot(init_vel);
            init_PE = PDE.V(init_pos).value();
            init_energy = init_KE + init_PE;
            find_path(init_pos, init_vel, max_t);
        }
    }

    void operator()(const Vector6d &state, Vector6d &dstate, double _) {
        auto E_val = PDE.E(state.head<3>());
        if (!E_val.has_value()) {
            throw std::runtime_error("Particle hit boundary");
        }
        dstate.head<3>() = state.tail<3>();
        dstate.tail<3>() = E_val.value() / mass;
    }

  private:
    const OctoTree &mesh_tree;
    const SolvePDE &PDE;
    const double mass;
    const bool record_path;
    std::optional<Vector3d> prev_pos1;
    std::optional<Vector3d> prev_pos2;

    std::pair<double, double> adjust_energy(Vector6d &state, const size_t max_iter = 100, const double rel_error = 1e-3) {
        const double tol = rel_error * init_energy;
        double KE, V;
        for (size_t iter = 0; iter < max_iter; ++iter) {
            const Vector3d pos = state.head<3>();
            const Vector3d vel = state.tail<3>();

            const auto [pot, E] = PDE.VE(pos).value();
            V = pot;
            KE = (mass / 2) * vel.squaredNorm();
            const double current_energy = KE + V;
            const double energy_error = current_energy - init_energy;

            if (std::abs(energy_error) < tol)
                return {KE, V};

            const double grad_norm_sq = E.squaredNorm() + 2 * mass * KE;
            const double scale = energy_error / grad_norm_sq;

            state.head<3>() += scale * E;
            state.tail<3>() -= scale * mass * vel;
        }

        return {KE, V};
    }

    bool check_collision(
        const Vector3d &p1,
        const Vector3d &p2,
        const Vector3d &v1,
        const Vector3d &v2,
        const Vector3d &v3,
        const double thresh = 1e-6) {
        Vector3d d = p2 - p1;
        Vector3d e1 = v2 - v1;
        Vector3d e2 = v3 - v1;

        Vector3d h = d.cross(e2);
        double a = e1.dot(h);
        if (std::abs(a) < thresh)
            return false;

        Vector3d s = p1 - v1;
        double u = s.dot(h) / a;
        if (u < 0.0 || u > 1.0)
            return false;

        Vector3d q = s.cross(e1);
        double v = d.dot(q) / a;
        if (v < 0.0 || (u + v) > 1.0)
            return false;

        double t = e2.dot(q) / a;
        return t >= 0.0 && t <= 1.0;
    }

    void observer(Vector6d &state, double time) {
        const Vector3d cur_pos = state.head<3>();
        const Vector3d cur_vel = state.tail<3>();
        double ke = (mass / 2) * state.tail<3>().squaredNorm();
        double pe = PDE.V(state.head<3>()).value();
        const double energy = ke + pe;
        const double rel_err = std::abs(energy / init_energy - 1);
        if (rel_err > 0.05) {
            const auto res = adjust_energy(state);
            ke = res.first;
            pe = res.second;
        }

        if (prev_pos1.has_value()) {
            const auto triangles = mesh_tree.query_w_radius((cur_pos + prev_pos1.value()) / 2, (cur_pos - prev_pos1.value()).norm() / 2);
            if (triangles.has_value()) {
                for (const auto &tri : triangles.value()) {
                    if (check_collision(cur_pos, prev_pos1.value(), mesh_tree.points[tri[0]], mesh_tree.points[tri[1]], mesh_tree.points[tri[2]])) {
                        throw std::runtime_error("Particle hit boundary");
                    }
                }
            }
            if (prev_pos2.has_value()) {
                const double n0 = cur_pos.norm();
                const double n1 = prev_pos1->norm();
                const double n2 = prev_pos2->norm();
                if (n0 < n1 && n2 < n1)
                    orbit_cnt++;
            }
        }

        if (record_path) {
            path_info.path.push_back(state);
            path_info.times.push_back(time);
            path_info.energies.emplace_back(ke, pe);
        }

        prev_pos2 = prev_pos1;
        prev_pos1 = cur_pos;
    }

    void find_path(
        const Vector3d &init_pos,
        const Vector3d &init_vel,
        const double max_t,
        const double init_dt = 1,
        const double abs_tol = 1e-4,
        const double rel_tol = 1e-4) {
        Vector6d state;
        state.head<3>() = init_pos;
        state.tail<3>() = init_vel;

        auto observer_wraper = [this](Vector6d &state, double time) {
            observer(state, time);
        };
        auto stepper = odeint::make_controlled<odeint::runge_kutta_dopri5<Vector6d>>(abs_tol, rel_tol);

        try {
            odeint::integrate_adaptive(stepper, *this, state, 0.0, max_t, init_dt, observer_wraper);
            sucsessful = true;
            eternal = true;
        } catch (const std::runtime_error &e) {
            if (std::string(e.what()) == "Particle hit boundary") {
                sucsessful = true;
                eternal = false;
            } else {
                std::cerr << "\nIntegration failed: " << e.what() << std::endl;
            }
        } catch (const std::exception &e) {
            std::cerr << "\nIntegration failed: " << e.what() << std::endl;
        }
    }
};

#endif
