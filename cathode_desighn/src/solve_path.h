#ifndef SOLVE_PATH_H
#define SOLVE_PATH_H

#include "octo_tree.h"
#include "solve_pde.h"
#include <Eigen/Dense>
#include <boost/numeric/odeint.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>

namespace odeint = boost::numeric::odeint;
using Vector3d = Eigen::Vector3d;
using Vector6d = Eigen::Vector<double, 6>;

class SolvePath {
  public:
    std::vector<Vector6d> path;
    std::vector<double> times;
    std::vector<std::pair<double, double>> energies;
    size_t orbit_cnt = 0;

    SolvePath(const std::unique_ptr<OctoTree> &mesh_tree_, const SolvePDE &PDE_, double mass_, const Vector3d &init_pos,
              const Vector3d &init_vel, const double max_t = 1e3)
        : mesh_tree(*mesh_tree_), PDE(PDE_), mass(mass_) {
        init_energy = (0.5 * mass) * init_vel.dot(init_vel) + PDE.V(init_pos).value();
        find_path(init_pos, init_vel, max_t);
    }

    void operator()(const Vector6d &state, Vector6d &dstate, double time) {
        auto E_val = PDE.E(state.head<3>());
        if (!E_val.has_value()) {
            path.push_back(state);
            times.push_back(time);
            std::cout << "Collision Found by Out Of Bounds Eval" << std::endl;
            throw std::runtime_error("Particle hit boundary");
        }
        dstate.head<3>() = state.tail<3>();
        dstate.tail<3>() = E_val.value() / mass;
    }

  private:
    const OctoTree &mesh_tree;
    const SolvePDE &PDE;
    const double mass;
    double init_energy;
    std::optional<std::vector<std::array<size_t, 3>>> prev_triangles;

    double KE(const Vector3d &vel) { return (mass / 2) * vel.squaredNorm(); }

    // unsafe
    double PE(const Vector3d &pos) { return PDE.V(pos).value(); }

    void adjust_energy(Vector6d &state, const size_t max_iter = 100, const double rel_error = 1e-3) {
        const double tol = rel_error * init_energy;
        for (size_t iter = 0; iter < max_iter; ++iter) {
            const Vector3d pos = state.head<3>();
            const Vector3d vel = state.tail<3>();

            const Vector3d E_field = PDE.E(pos).value();
            const double current_energy = KE(vel) + PE(pos);
            const double energy_error = current_energy - init_energy;

            if (std::abs(current_energy - init_energy) < tol)
                break;

            const double grad_norm_sq = E_field.squaredNorm() + mass * mass * vel.squaredNorm();
            const double scale = energy_error / grad_norm_sq;

            state.head<3>() += scale * E_field;
            state.tail<3>() -= scale * mass * vel;
        }
    }

    bool check_collision(const Vector3d &p1, const Vector3d &p2, const Vector3d &v1, const Vector3d &v2,
                         const Vector3d &v3, const double thresh = 1e-6) {

        Vector3d min_tri = v1.cwiseMin(v2).cwiseMin(v3);
        Vector3d max_tri = v1.cwiseMax(v2).cwiseMax(v3);
        Vector3d min_seg = p1.cwiseMin(p2);
        Vector3d max_seg = p1.cwiseMax(p2);

        if ((max_seg.array() < min_tri.array()).any() || (min_seg.array() > max_tri.array()).any()) {
            return false;
        }

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
        double ke = KE(state.tail<3>());
        double pe = PE(state.head<3>());
        const double energy = ke + pe;
        const double rel_err = std::abs(energy / init_energy - 1);
        if (rel_err > 0.05) {
            adjust_energy(state);
            ke = KE(state.tail<3>());
            pe = PE(state.head<3>());
        }

        const auto cur_triangles = mesh_tree.query(cur_pos);
        if (!path.empty()) {
            std::set<std::array<size_t, 3>> unique_triangles;
            if (cur_triangles.has_value()) {
                unique_triangles.insert(cur_triangles->begin(), cur_triangles->end());
            }
            if (prev_triangles.has_value()) {
                unique_triangles.insert(prev_triangles->begin(), prev_triangles->end());
            }

            for (const auto &tri : unique_triangles) {
                if (check_collision(path.back().head<3>(), cur_pos, mesh_tree.points[tri[0]], mesh_tree.points[tri[1]],
                                    mesh_tree.points[tri[2]])) {
                    path.push_back(state);
                    times.push_back(time);
                    std::cout << "Collision Found by Octo-Tree" << std::endl;
                    throw std::runtime_error("Particle hit boundary");
                }
            }
        }
        if (path.size() > 1) {
            const double n0 = cur_pos.norm();
            const double n1 = path.back().head<3>().norm();
            const double n2 = path[path.size() - 2].head<3>().norm();
            if (n0 < n1 && n2 < n1)
                orbit_cnt++;
        }
        path.push_back(state);
        times.push_back(time);
        energies.emplace_back(ke, pe);
        prev_triangles = cur_triangles;
    }

    void find_path(const Vector3d &init_pos, const Vector3d &init_vel, const double max_t, const double init_dt = 1,
                   const double abs_tol = 1e-4, const double rel_tol = 1e-4) {
        Vector6d state;
        state.head<3>() = init_pos;
        state.tail<3>() = init_vel;

        path.clear();
        path.reserve(1000);
        auto observer_wraper = [this](Vector6d &state, double time) { observer(state, time); };
        auto stepper = odeint::make_controlled<odeint::runge_kutta_dopri5<Vector6d>>(abs_tol, rel_tol);

        try {
            odeint::integrate_adaptive(stepper, *this, state, 0.0, max_t, init_dt, observer_wraper);
            std::cout << "No Collison" << std::endl;
        } catch (const std::runtime_error &e) {
            if (std::string(e.what()) == "Particle hit boundary") {
                std::cout << "Collision Detected at " << times.back() << " ns" << std::endl;
            } else {
                std::cerr << "Integration failed: " << e.what() << std::endl;
            }
        } catch (const std::exception &e) {
            std::cerr << "Integration failed: " << e.what() << std::endl;
        }
    }
};

#endif
