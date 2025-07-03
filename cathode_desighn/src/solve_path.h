#ifndef SOLVE_PATH_H
#define SOLVE_PATH_H

#include "octo_tree.h"
#include "solve_pde.h"
#include <Eigen/Dense>
#include <boost/numeric/odeint.hpp>
#include <cstddef>
#include <cstdlib>
#include <utility>

template <size_t N> class SolveMultiPath {
    using Vector3d = Eigen::Vector3d;
    using Vector6d = Eigen::Vector<double, 6>;
    using Stepper =
        boost::numeric::odeint::controlled_runge_kutta<boost::numeric::odeint::runge_kutta_dopri5<Vector6d>>;

  private:
    struct PathInfo {
        std::vector<Vector3d> positions{};
        std::vector<Vector3d> velocities{};
        std::vector<double> times{};
        std::vector<std::pair<double, double>> energies{};

        PathInfo(const size_t size = 1 << 8) {
            positions.reserve(size);
            velocities.reserve(size);
            times.reserve(size);
            energies.reserve(size);
        }
    };

    struct Path {
        Stepper stepper;
        Vector3d pos;
        Vector3d vel;
        double time = 0.0;
        double dt = 0.001;
        double V;
        Vector3d E;

        bool active = true;
        double init_energy;
        size_t orbit_cnt = 0;
        Vector3d prev_pos = pos;
        double prev_prev_norm = pos.norm();
        std::optional<PathInfo> path_info;

        Path() {
            pos = Vector3d::Zero();
            vel = Vector3d::Zero();
            E = Vector3d::Zero();
        }

        Path(const Vector3d &init_pos, const Vector3d &init_vel) {
            pos = init_pos;
            vel = init_vel;
        }

        Path(
            const Vector3d &init_pos,
            const Vector3d &init_vel,
            const double init_ke,
            const double init_pe,
            const Vector3d &init_E,
            const bool record_path,
            const double abs_err = 1e-4,
            const double rel_err = 1e-4)
            : stepper(boost::numeric::odeint::make_controlled<boost::numeric::odeint::runge_kutta_dopri5<Vector6d>>(
                  abs_err,
                  rel_err)),
              pos(init_pos), vel(init_vel), time(0.0), dt(0.001), V(init_pe), E(init_E),
              init_energy(init_ke + init_pe) {
            if (record_path) {
                path_info = PathInfo();
            }
        }

        bool update(const double mass, const double max_t) {
            Vector6d state;
            state.head<3>() = pos;
            state.tail<3>() = vel;
            auto system = [&](const Vector6d &state, Vector6d &dstate, double _) {
                dstate.head<3>() = state.tail<3>();
                dstate.tail<3>() = E / mass;
            };
            dt = std::min(dt, max_t - time);
            auto result = stepper.try_step(system, state, time, dt);
            while (result == boost::numeric::odeint::controlled_step_result::fail) {
                dt = std::min(dt, max_t - time);
                result = stepper.try_step(system, state, time, dt);
            }

            prev_prev_norm = prev_pos.norm();
            prev_pos = pos;
            pos = state.head<3>();
            vel = state.tail<3>();
            if (path_info.has_value()) {
                path_info->positions.push_back(pos);
                path_info->velocities.push_back(vel);
                path_info->times.push_back(time);
                path_info->energies.push_back(std::make_pair(0.5 * mass * vel.squaredNorm(), V));
            }
            return time < max_t;
        }

        void print() {
            std::cout << "Path: (" << time << " s)" << std::endl;
            std::cout << "  pos: " << pos.transpose() << " , vel: " << vel.transpose() << std::endl;
            std::cout << "  V: " << V << " , E: " << E.transpose() << std::endl;
        }
    };

    const SolvePDE<N> &pde;
    const OctoTree &mesh_tree;
    const double mass;
    const double max_time;
    std::array<Path, N> paths;

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

    void adjust_energies(
        std::vector<bool> &adjust_energy_paths,
        const size_t max_iter = 10,
        const double rel_tol = 1e-5) {
        std::vector<Vector3d> points;
        std::vector<size_t> path_idxs;
        for (size_t iter = 0; iter < max_iter; iter++) {
            points.clear();
            path_idxs.clear();
            for (size_t i = 0; i < N; i++) {
                if (adjust_energy_paths[i]) {
                    const double ke = (mass / 2) * paths[i].vel.squaredNorm();
                    const double scale = (ke + paths[i].V - paths[i].init_energy) /
                                         (paths[i].E.squaredNorm() + 2 * mass * ke);
                    paths[i].pos += scale * paths[i].E;
                    paths[i].vel *= (1 - scale * mass);
                    points.push_back(paths[i].pos);
                    path_idxs.push_back(i);
                }
            }

            auto VE_vals = pde.multi_VE(points, path_idxs);
            for (size_t i = 0; i < VE_vals.size(); i++) {
                if (VE_vals[i].has_value()) {
                    paths[path_idxs[i]].V = VE_vals[i]->first;
                    paths[path_idxs[i]].E = VE_vals[i]->second;
                } else {
                    paths[path_idxs[i]].active = false;
                    adjust_energy_paths[path_idxs[i]] = false;
                }
            }

            for (size_t i = 0; i < N; i++) {
                if (adjust_energy_paths[i]) {
                    const double energy = (mass / 2) * paths[i].vel.squaredNorm() + paths[i].V;
                    if (std::abs(energy / paths[i].init_energy - 1) < rel_tol) {
                        adjust_energy_paths[i] = false;
                    }
                }
            }
        }
    }

    void integrate(const double energy_rel_tol = 0.05) {
        while (true) {
            for (size_t i = 0; i < N; i++) {
                if (paths[i].active) {
                    if (!paths[i].update(mass, max_time)) {
                        paths[i].active = false;
                    }
                }
            }

            for (size_t i = 0; i < N; i++) {
                if (paths[i].active) {
                    const auto &p1 = paths[i].pos;
                    const auto &p2 = paths[i].prev_pos;
                    const auto triangles = mesh_tree.query_w_radius((p1 + p2) / 2, (p1 - p2).norm() / 2);
                    if (triangles.has_value()) {
                        for (const auto &tri : triangles.value()) {
                            if (check_collision(
                                    p1,
                                    p2,
                                    mesh_tree.points[tri[0]],
                                    mesh_tree.points[tri[1]],
                                    mesh_tree.points[tri[2]])) {
                                paths[i].active = false;
                            }
                        }
                    }
                }
            }

            std::vector<Vector3d> points;
            std::vector<size_t> path_idxs;
            for (size_t i = 0; i < N; i++) {
                if (paths[i].active) {
                    points.push_back(paths[i].pos);
                    path_idxs.push_back(i);
                }
            }
            auto VE_vals = pde.multi_VE(points, path_idxs);
            for (size_t i = 0; i < VE_vals.size(); i++) {
                if (VE_vals[i].has_value()) {
                    paths[path_idxs[i]].V = VE_vals[i]->first;
                    paths[path_idxs[i]].E = VE_vals[i]->second;
                } else {
                    paths[path_idxs[i]].active = false;
                }
            }

            std::vector<bool> adjust_energy_paths(N, false);
            for (size_t i = 0; i < N; i++) {
                if (paths[i].active) {
                    const double energy = (mass / 2) * paths[i].vel.squaredNorm() + paths[i].V;
                    if (std::abs(energy / paths[i].init_energy - 1) > energy_rel_tol) {
                        adjust_energy_paths[i] = true;
                    }
                }
            }

            adjust_energies(adjust_energy_paths);
        }
    }

  public:
    SolveMultiPath(
        const OctoTree &mesh_tree_,
        const SolvePDE<N> &pde_,
        const double mass_,
        const std::array<Vector3d, N> &init_pos,
        const std::array<Vector3d, N> &init_vel,
        const double max_time_ = 1e4,
        const size_t record_density = N)
        : pde(pde_), mesh_tree(mesh_tree_), mass(mass_), max_time(max_time_) {
        auto valid_paths = pde.init_cache(init_pos);
        for (size_t i = 0; i < N; i++) {
            if (valid_paths[i].has_value()) {
                const auto &[V, E] = valid_paths[i].value();
                const double ke = (mass / 2) * init_vel[i].squaredNorm();
                paths[i] = Path(init_pos[i], init_vel[i], ke, V, E, i % record_density == 0);
            } else {
                paths[i] = Path(init_pos[i], init_vel[i]);
            }
        }

        integrate();
    }
};

#endif
