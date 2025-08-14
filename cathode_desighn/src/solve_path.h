#ifndef SOLVE_PATH_H
#define SOLVE_PATH_H

#include "solve_pde.h"
#include <Eigen/Dense>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/stepper/symplectic_euler.hpp>
#include <boost/numeric/odeint/stepper/symplectic_rkn_sb3a_m4_mclachlan.hpp>
#include <deal.II/lac/vector.h>
#include <optional>
#include <stdexcept>
#include <utility>

namespace odeint = boost::numeric::odeint;
using Vector3d = Eigen::Vector3d;
class SolvePath {
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

		void add(const std::optional<std::pair<Vector3d, Vector3d>> state, const std::optional<double> ke,
				 const std::optional<double> pe, const std::optional<double> time) {
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
				KE.push_back(KE.back() + PE.back() - pe.value());
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
	size_t energy_correction_cnt;

	std::optional<double> prev_prev_norm;
	std::optional<double> prev_norm;

	void correct_energy(std::pair<Vector3d, Vector3d> &state, double &V, Vector3d &E) {
		constexpr double rel_energy_tol = 1e-5;
		constexpr size_t max_iter = 10;
		const double abs_energy_tol = std::abs(rel_energy_tol * init_energy);
		auto [q, p] = state;
		for (size_t i = 0; i < max_iter; i++) {
			const double p2 = p.squaredNorm();
			const double energy_diff = p2 / (2 * mass) + V - init_energy;
			if (std::abs(energy_diff) < abs_energy_tol) {
				state.first = q;
				state.second = p;
				return;
			}
			const double scale = energy_diff / (E.squaredNorm() + p2 / (mass * mass));
			Vector3d new_q, new_p;

			double size = 1;
			constexpr double SIZE_REDUCTION = 0.5;
			constexpr double MAX_RETRY = 5;
			for (size_t i = 0; i <= MAX_RETRY; i++) {
				new_q = q + scale * size * E;
				new_p = p * (1 - size * scale / mass);
				auto opt_VE = PDE_SOL.VE(new_q);
				if (opt_VE.has_value()) {
					V = opt_VE.value().first;
					E = opt_VE.value().second;
					q = new_q;
					p = new_p;
					break;
				}
				size *= SIZE_REDUCTION;
			}
		}
		state.first = q;
		state.second = p;
	}

	void observe(std::pair<Vector3d, Vector3d> &state, const double t) {
		constexpr double ENERGY_REL_TOL = 0.05;
		constexpr size_t ENERGY_CORECTION_FR = 100;
		if (energy_correction_cnt % ENERGY_CORECTION_FR == 0) {
			auto VE_opt = PDE_SOL.VE(state.first);
			if (VE_opt.has_value()) {
				auto [V, E] = VE_opt.value();
				const double energy = V + state.second.squaredNorm() / (2 * mass);
				const double energy_rel_err = std::abs(energy / init_energy - 1);
				if (energy_rel_err > ENERGY_REL_TOL) {
					correct_energy(state, V, E);
				}
			} else {
				return;
			}
		}
		energy_correction_cnt++;

		const double cur_norm = state.first.norm();
		if (prev_norm.has_value() && prev_prev_norm.has_value()) {
			if (cur_norm < prev_norm.value() && prev_prev_norm.value() < prev_norm.value()) {
				orbit_cnt++;
			}
		}
		prev_prev_norm = prev_norm;
		prev_norm = cur_norm;

		if (record_path) {
			path_info.add(state, state.second.squaredNorm() / (2 * mass), std::nullopt, t);
		}
	}

	void find_path(const Vector3d &init_q, const Vector3d &init_p) {
		constexpr int MAX_RETRY = 5;
		constexpr double STEP_REDUCTION = 0.5;
		constexpr double STEP_INCREASE = 1.5;
		constexpr double MIN_DT = 1e-3;

		double t = 0.0;
		double dt = max_dt;
		auto state = std::make_pair(init_q, init_p);

		bool oob = false;
		auto system = std::make_pair([&](auto const &p, auto &dq) { dq = p / mass; },
									 [&](auto const &q, auto &dp) {
										 auto e = PDE_SOL.E(q);
										 if (!e) {
											 oob = true;
											 dp.setZero();
										 } else
											 dp = *e;
									 });

		odeint::symplectic_rkn_sb3a_m4_mclachlan<Vector3d> stepper;
		try {
			while (true) {
				oob = false;
				const auto cur_state = state;
				for (int retry_cnt = 0; retry_cnt <= MAX_RETRY; retry_cnt++) {
					stepper.do_step(system, state, t, dt);
					if (!oob) {
						dt = std::min(dt * STEP_INCREASE, max_dt);
						break;
					} else {
						if (retry_cnt == MAX_RETRY) {
							sucsessful = true;
							return;
						} else {
							state = cur_state;
							dt = std::min(dt * STEP_REDUCTION, MIN_DT);
						}
					}
				}
				t += dt;
				observe(state, t);
			}
		} catch (std::runtime_error &e) {
			sucsessful = false;
			std::cerr << "Runtime Erorr: " << e.what() << std::endl;
		}
	}

  public:
	double init_energy;
	double init_KE;
	double init_PE;
	PathInfo path_info;
	bool sucsessful = false;
	size_t orbit_cnt = 0;

	SolvePath(const SolvePDE &PDE_SOL, const double mass, const Vector3d init_pos, const Vector3d init_mom,
			  const double max_dt = 0.125, const bool record_path = false)
		: PDE_SOL(PDE_SOL), mass(mass), record_path(record_path), max_dt(max_dt) {
		if (PDE_SOL.init_cache(init_pos)) {
			init_KE = init_mom.squaredNorm() / (2 * mass);
			init_PE = PDE_SOL.V(init_pos).value();
			init_energy = init_KE + init_PE;
			std::pair<Vector3d, Vector3d> init_state;
			if (record_path) {
				path_info = PathInfo(1000);
				path_info.add(init_state, init_KE, init_PE, 0);
			}
			find_path(init_pos, init_mom);
		}
	}
};

#endif
