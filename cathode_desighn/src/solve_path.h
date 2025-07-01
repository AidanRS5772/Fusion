#ifndef SOLVE_PATH_H
#define SOLVE_PATH_H

#include "octo_tree.h"
#include "solve_pde.h"
#include <Eigen/Dense>
#include <boost/numeric/odeint.hpp>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <unordered_set>
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
    };

    struct Path {
        Stepper stepper;
        Vector3d pos;
        Vector3d vel;
        double time = 0.0;
        double dt = 0.001;
        double V;
        Vector3d E;
        PathInfo path_info;

        const size_t id;
        const double init_energy;
        const bool record_path;
        size_t orbit_cnt = 0;
        Vector3d prev_pos = pos;
        double prev_prev_norm = pos.norm();

        Path(
            const Vector3d &init_pos,
            const Vector3d &init_vel,
            const double init_ke,
            const double init_pe,
            const Vector3d &init_E,
            const size_t id_,
            const bool record_path_,
            const double abs_err = 1e-4,
            const double rel_err = 1e-4)
            : stepper(boost::numeric::odeint::make_controlled<boost::numeric::odeint::runge_kutta_dopri5<Vector6d>>(
                  abs_err,
                  rel_err)),
              pos(init_pos), vel(init_vel), time(0.0), dt(0.001), V(init_pe), E(init_E), id(id_),
              init_energy(init_ke + init_pe), record_path(record_path_) {
            if (record_path) {
                path_info.positions.reserve(1000);
                path_info.velocities.reserve(1000);
                path_info.times.reserve(1000);
                path_info.energies.reserve(1000);
                path_info.positions.push_back(init_pos);
                path_info.velocities.push_back(init_vel);
                path_info.times.push_back(0.0);
                path_info.energies.push_back(std::make_pair(init_ke, init_pe));
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
            stepper.try_step(system, state, time, dt);
            dt = std::min(dt, max_t - time);
            auto result = stepper.try_step(system, state, time, dt);
            while (result == boost::numeric::odeint::controlled_step_result::fail) {
                result = stepper.try_step(system, state, time, dt);
            }
            prev_prev_norm = prev_pos.norm();
            prev_pos = pos;
            pos = state.head<3>();
            vel = state.tail<3>();
            if (record_path) {
                path_info.positions.push_back(pos);
                path_info.velocities.push_back(vel);
                path_info.times.push_back(time);
                path_info.energies.push_back(std::make_pair(0.5 * mass * vel.squaredNorm(), V));
            }
            return time < max_t;
        }

        void print() {
            std::cout << "Path: " << id << " (" << time << " s)" << std::endl;
            std::cout << "  pos: " << pos.transpose() << " , vel: " << vel.transpose() << std::endl;
            std::cout << "  V: " << V << " , E: " << E.transpose() << std::endl;
        }
    };

    const SolvePDE &pde;
    const OctoTree &mesh_tree;
    const double mass;
    const double max_time;
    std::unordered_set<std::unique_ptr<Path>> paths;
    size_t active_path_cnt;

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
        for (size_t i = 0; i < max_iter; i++) {
            std::vector<Vector3d> points;
            points.reserve(active_path_cnt);
            std::vector<size_t> path_ids;
            path_ids.reserve(active_path_cnt);
            for (size_t j = 0; j < active_path_cnt; j++) {
                if (adjust_energy_paths[j]) {
                    const double ke = (mass / 2) * paths[j]->vel.squaredNorm();
                    const double scale = (ke + paths[j]->V - paths[j]->init_energy) /
                                         (paths[j]->E.squaredNorm() + 2 * mass * ke);
                    paths[j]->pos += scale * paths[j]->E;
                    paths[j]->vel *= (1 - scale * mass);

                    points.push_back(paths[j]->pos);
                    path_ids.push_back(paths[j]->id);
                }
            }

            auto VE_vals = pde.multi_VE(points, path_ids);
            for (size_t i = points.size() - 1; i > VE_vals.size(); i++) {
                const double energy = (mass / 2) * paths[path_ids[i]]->vel.squaredNorm() + paths[path_ids[i]]->V;
                if (std::abs(energy / paths[path_ids[i]]->init_energy - 1) < rel_tol) {
                    adjust_energy_paths[path_ids[i]] = false;
                }
            }

            bool flag = false;
            for (size_t i = 0; i < active_path_cnt; i++) {
                if (adjust_energy_paths[i]) {
                    const double energy = (mass / 2) * paths[i]->vel.squaredNorm() + paths[i]->V;
                    if (std::abs(energy / paths[i]->init_energy - 1) < rel_tol) {
                        adjust_energy_paths[i] = false;
                    }
                    flag = true;
                }
            }
            if (flag) {
                break;
            }
        }
    }

    void integrate(const double energy_rel_tol = 0.05) {
        while (active_path_cnt > 0) {
            for (size_t i = 0; i < active_path_cnt; i++) {
                if (!paths[i]->update(mass, max_time)) {
                    std::swap(paths[i + 1], paths[--active_path_cnt]);
                }
            }

            for (size_t i = 0; i < active_path_cnt; i++) {
                const auto &p1 = paths[i]->pos;
                const auto &p2 = paths[i]->prev_pos;
                const auto triangles = mesh_tree.query_w_radius((p1 + p2) / 2, (p1 - p2).norm() / 2);
                if (triangles.has_value()) {
                    for (const auto &tri : triangles.value()) {
                        if (check_collision(
                                p1,
                                p2,
                                mesh_tree.points[tri[0]],
                                mesh_tree.points[tri[1]],
                                mesh_tree.points[tri[2]])) {
                            std::swap(paths[i + 1], paths[--active_path_cnt]);
                        }
                    }
                }
            }

            std::vector<Vector3d> points(active_path_cnt);
            std::vector<size_t> path_ids(active_path_cnt);
            for (size_t i = 0; i < active_path_cnt; i++) {
                points[i] = paths[i]->pos;
                path_ids[i] = paths[i]->id;
            }

            auto VE_vals = pde.multi_VE(points, path_ids);
            for (int i = active_path_cnt - 1; i >= 0; i--) {
                if (VE_vals[i].has_value()) {
                    paths[i]->V = VE_vals[i]->first;
                    paths[i]->E = VE_vals[i]->second;
                } else {
                    std::swap(paths[i], paths[--active_path_cnt]);
                }
            }

            std::vector<bool> adjust_energy_paths(active_path_cnt, false);
            for (size_t i = 0; i < active_path_cnt; i++) {
                const double energy = (mass / 2) * paths[i]->vel.squaredNorm() + paths[i]->V;
                if (std::abs(energy / paths[i]->init_energy - 1) > energy_rel_tol) {
                    adjust_energy_paths[i] = true;
                }
            }

            adjust_energies(adjust_energy_paths);
        }
    }

  public:
    SolveMultiPath(
        const OctoTree &mesh_tree_,
        const SolvePDE &pde_,
        const double mass_,
        const std::array<Vector3d, N> &init_pos,
        const std::array<Vector3d, N> &init_vel,
        const double max_time_ = 1e4,
        const size_t record_density = N)
        : pde(pde_), mesh_tree(mesh_tree_), mass(mass_), max_time(max_time_) {
        auto valid_paths = pde.init_cache(init_pos);
        active_path_cnt = 0;
        for (size_t i = 0; i < N; i++) {
            if (valid_paths[i].has_value()) {
                const auto &[V, E] = valid_paths[i].value();
                const double ke = (mass / 2) * init_vel[i].squaredNorm();
                paths[active_path_cnt] = std::make_unique<Path>(
                    init_pos[i],
                    init_vel[i],
                    ke,
                    V,
                    E,
                    active_path_cnt,
                    active_path_cnt % record_density == 0);
                active_path_cnt++;
            }
        }

        integrate();
    }
};

#endif
