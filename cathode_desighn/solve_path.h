#ifndef SOLVE_PATH_H
#define SOLVE_PATH_H

#include <boost/numeric/odeint.hpp>
#include <functional>

namespace odeint = boost::numeric::odeint;

class SolvePath {
  public:
    using V_func = std::function<double(const std::array<double, 3> &)>;
    using E_func = std::function<std::array<double, 3>(const std::array<double, 3> &)>;

    std::vector<std::array<double, 7>> path;
    bool success;

    SolvePath(V_func V_, E_func E_, double charge_, double mass_, const std::array<double, 3> &init_pos,
              const std::array<double, 3> &init_vel)
        : V(V_), E(E_), charge(charge_), mass(mass_) {
        find_path(init_pos, init_vel);
    }

    void operator()(const std::array<double, 6> &state, std::array<double, 6> &dstate, double _) {
        std::array<double, 3> E_val = E({state[0], state[1], state[2]});
        dstate = {state[3], state[4], state[5], qm * E_val[0], qm * E_val[1], qm * E_val[2]};
    }

  private:
    V_func V;
    E_func E;
    const double charge;
    const double mass;
    const double qm = charge / mass;

    void find_path(const std::array<double, 3> &init_pos, const std::array<double, 3> &init_vel,
                   const double max_t = 1e3, const double init_dt = 1e-3, const double abs_tol = 1e-6,
                   const double rel_tol = 1e-4) {
        std::array<double, 6> state = {init_pos[0], init_pos[1], init_pos[2],
                                       init_vel[0], init_vel[1], init_vel[2]};

        path.clear();
        path.reserve(1000);
        auto observer = [this](const std::array<double, 6> state, double time) {
            path.push_back({state[0], state[1], state[2], state[3], state[4], state[5], time});
        };

        auto stepper =
            odeint::make_controlled<odeint::runge_kutta_dopri5<std::array<double, 6>>>(abs_tol, rel_tol);

        try {
            odeint::integrate_adaptive(stepper, *this, state, 0.0, max_t, init_dt, observer);
            success = true;
            std::cout << "Integration completed. Points: " << path.size()
                      << ", Final time: " << path.back()[6] << " ns" << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Integration failed: " << e.what() << std::endl;
            success = false;
            return;
        }
    }
};

#endif
