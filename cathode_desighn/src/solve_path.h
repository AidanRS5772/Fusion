#ifndef SOLVE_PATH_H
#define SOLVE_PATH_H

#include "solve_pde.h"
#include <Eigen/Dense>
#include <bit>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/stepper/symplectic_euler.hpp>
#include <boost/numeric/odeint/stepper/symplectic_rkn_sb3a_m4_mclachlan.hpp>
#include <cmath>
#include <cstdint>
#include <deal.II/lac/vector.h>
#include <limits>
#include <optional>
#include <utility>

namespace odeint = boost::numeric::odeint;
using Vector3d = Eigen::Vector3d;
class SolvePath {

	struct PathInfo {
		Vector3d init_q = Vector3d::Zero();
		Vector3d init_p = Vector3d::Zero();
		double fusion_prop = 0;
		double char_time = 0;
		double init_KE = 0;
		double init_PE = 0;
		double init_E = 0;
		size_t orbit_cnt = 0;

		bool active = false;
		std::vector<std::pair<Vector3d, Vector3d>> states{};
		std::vector<double> times{};
		std::vector<double> KE{};
		std::vector<double> PE{};

		PathInfo(const size_t reserve_size = 1000) {
			states.reserve(reserve_size);
			KE.reserve(reserve_size);
			PE.reserve(reserve_size);
		};

		void add(const std::optional<std::pair<Vector3d, Vector3d>> state,
		         const std::optional<double> ke,
		         const std::optional<double> pe,
		         const std::optional<double> time) {
			if (active) {
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
		}
	};

	struct FeildCache {
		const SolvePDE &PDE_SOL;
		Vector3d qV = Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
		Vector3d qE = Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
		Vector3d E = Vector3d::Zero();
		double V = 0;

		FeildCache(const SolvePDE &pde_sol) : PDE_SOL(pde_sol) {}

		bool init_cache(const Vector3d &q) { return PDE_SOL.init_cache(q); }

		bool V_eval(const Vector3d &q) {
			if (q.isApprox(qV)) return true;
			qV = q;
			return PDE_SOL.V(q, V);
		}

		bool E_eval(const Vector3d &q) {
			if (q.isApprox(qE)) return true;
			qE = q;
			return PDE_SOL.E(q, E);
		}

		bool VE_eval(const Vector3d &q) {
			const bool E_aprx = q.isApprox(qE);
			const bool V_aprx = q.isApprox(qV);
			if (E_aprx && V_aprx) {
				return true;
			} else if (!E_aprx && V_aprx) {
				qE = q;
				return PDE_SOL.E(q, E);
			} else if (E_aprx && !V_aprx) {
				qV = q;
				return PDE_SOL.V(q, V);
			} else {
				qE = q;
				qV = q;
				return PDE_SOL.VE(q, V, E);
			}
		}
	};

	static constexpr size_t POLY_REG_ORDER = 4;

	const double m, V, r, R, T;

	FeildCache VE_cache;
	double prv = std::numeric_limits<double>::quiet_NaN();
	size_t energy_correction_cnt;

	bool reset(const Vector3d &q, const Vector3d &p, const bool record_path) {
		if (VE_cache.init_cache(q)) {
			path_info.init_q = q;
			path_info.init_p = p;
			path_info.char_time = characteristic_time(q.norm());
			path_info.init_KE = p.squaredNorm() / (2 * m);
			VE_cache.V_eval(q);
			path_info.init_PE = VE_cache.V;
			path_info.init_E = path_info.init_KE + path_info.init_PE;
			path_info.fusion_prop = fp(V + path_info.init_E);
			path_info.orbit_cnt = 0;

			path_info.active = record_path;
			path_info.states.clear();
			path_info.times.clear();
			path_info.KE.clear();
			path_info.PE.clear();

			energy_correction_cnt = 1;
			prv = std::numeric_limits<double>::quiet_NaN();

			return true;
		}
		return false;
	}

	double fusion_probability(const double K0) {
		const double Ke = V * (1 / r - 1.5 * (r + R) / (r * r + r * R + R * R)) / (1 / r - 1 / R) + 1.5 * T * 1e-6;
		constexpr double Eg = 1875.612928 * M_PI * 0.0072973525643;
		constexpr std::array<double, 5> a = {54.6385, 270.405, -79.849, 15.41285, -1.166645};

		boost::math::quadrature::exp_sinh<double> integrator;
		const double f_int = integrator.integrate([K0, a, Ke](const double E) {
			const double S = a[0] + E * (a[1] + E * (a[2] + E * (a[3] + a[4] * E)));
			return std::sqrt(E) * S * std::sinh(3 * std::sqrt(2 * K0 * E) / Ke)
			       * std::exp(-3 * E / Ke - std::sqrt(Eg / E));
		});

		const double den = std::sqrt(K0)
		                   + std::sqrt(3 * M_PI / (2 * Ke)) * (K0 + Ke / 3) * std::erf(std::sqrt(3 * K0 / (2 * Ke)))
		                         * std::exp(3 * K0 / (2 * Ke));
		return f_int / den;
	}

	template <size_t M>
	std::pair<Eigen::Matrix<double, POLY_REG_ORDER + 1, 1>, double>
	poly_reg(const std::function<double(double)> f, const double a, const double b) {
		Eigen::Matrix<double, M, 1> Y;
		Eigen::Matrix<double, M, POLY_REG_ORDER + 1> V;
		std::vector<double> X(M);
		for (size_t i = 0; i < M; i++) {
			const double x = a + i * (b - a) / (M - 1);
			Y(i) = f(x);

			double p = 1.0;
			for (size_t j = 0; j < POLY_REG_ORDER + 1; j++) {
				V(i, j) = p;
				p *= x;
			}
		}

		Eigen::Matrix<double, POLY_REG_ORDER + 1, 1> coefs = (V.transpose() * V).ldlt().solve(V.transpose() * Y);
		double res = std::sqrt((V * coefs - Y).squaredNorm() / M);
		return {coefs, res};
	}

	double exp_max_energy(const double sample_size) {
		constexpr double gamma = 0.577215664901532;
		return V
		       + 1e-6 * T
		             * (std::log(2) - std::log(M_PI) / 2 + gamma + std::log(sample_size)
		                + std::log(std::log(sample_size)));
	}

	double fp(const double x) const {
		double y = 0;
		for (int i = static_cast<int>(POLY_REG_ORDER); i >= 0; i--) y = y * x + poly_reg_coefs[i];
		return y;
	}

	double characteristic_time(const double r0) {
		const double mu = 1 / (1 / r - 1 / R);
		const double R0 = r / r0;
		const double inner_cathode = r * std::sqrt(r / (1 - R0));
		const double outer_cathode = r0 * std::sqrt(r0) * (std::acos(std::sqrt(R0)) + std::sqrt(R0 * (1 - R0)));
		return std::sqrt(2 * m / (mu * V)) * (inner_cathode + outer_cathode);
	}

	void correct_energy(std::pair<Vector3d, Vector3d> &state) {
		constexpr double rel_energy_tol = 1e-5;
		constexpr size_t max_iter = 10;
		const double abs_energy_tol = std::abs(rel_energy_tol * path_info.init_E);
		auto [q, p] = state;
		for (size_t i = 0; i < max_iter; i++) {
			const double p2 = p.squaredNorm();
			const double energy_diff = p2 / (2 * m) + VE_cache.V - path_info.init_E;
			if (std::abs(energy_diff) < abs_energy_tol) {
				state.first = q;
				state.second = p;
				return;
			}
			const double scale = energy_diff / (VE_cache.E.squaredNorm() + p2 / (m * m));
			Vector3d new_q, new_p;

			double size = 1;
			constexpr double SIZE_REDUCTION = 0.5;
			constexpr double MAX_RETRY = 5;
			for (size_t i = 0; i <= MAX_RETRY; i++) {
				new_q = q + scale * size * VE_cache.E;
				new_p = p * (1 - size * scale / m);
				if (VE_cache.VE_eval(new_q)) {
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

	void observe(std::pair<Vector3d, Vector3d> &state, const double t, const double max_dt) {
		constexpr double ENERGY_REL_TOL_LO = 0.01;
		constexpr double ENERGY_REL_TOL_HI = 0.05;
		constexpr size_t ENERGY_CORECTION_FR = 4;
		if (energy_correction_cnt % (static_cast<size_t>(path_info.char_time / max_dt) / ENERGY_CORECTION_FR) == 0)
		    [[unlikely]] {
			if (VE_cache.VE_eval(state.first)) {
				const double p_norm = state.second.norm();
				const double energy = VE_cache.V + p_norm * p_norm / (2 * m);
				const double energy_rel_err = std::abs(energy / path_info.init_E - 1);
				if (energy_rel_err > ENERGY_REL_TOL_HI) {
					correct_energy(state);
				} else if (energy_rel_err > ENERGY_REL_TOL_LO && p_norm > 1e-12) {
					state.second *= std::sqrt(2 * m * (path_info.init_E - VE_cache.V)) / state.second.norm();
				}
				path_info.add(state, state.second.squaredNorm() / (2 * m), V, t);
			} else {
				path_info.add(state, state.second.squaredNorm() / (2 * m), std::nullopt, t);
			}
		} else {
			path_info.add(state, state.second.squaredNorm() / (2 * m), std::nullopt, t);
		}
		energy_correction_cnt++;

		double rv = state.first.dot(state.second);
		if (prv * rv < 0 && state.first.norm() > r) {
			path_info.orbit_cnt++;
		}
		prv = rv;
	}

#if defined(__clang__)
#pragma clang optimize off
#endif
	inline bool is_not_valid(double x) noexcept { return ((std::bit_cast<uint64_t>(x) >> 52) & 0x7FF) == 0x7FF; }
#if defined(__clang__)
#pragma clang optimize on
#endif

	inline bool state_is_not_valid(const std::pair<Vector3d, Vector3d> &s) noexcept {
		return is_not_valid(s.first(0)) || is_not_valid(s.first(1)) || is_not_valid(s.first(2))
		       || is_not_valid(s.second(0)) || is_not_valid(s.second(1)) || is_not_valid(s.second(2));
	}

  public:
	PathInfo path_info;
	double poly_reg_res;
	std::array<double, POLY_REG_ORDER + 1> poly_reg_coefs;

	SolvePath(const SolvePDE &PDE_SOL,
	          const double mass,
	          const double voltage,
	          const double cathode_radius,
	          const double anode_radius,
	          const double temprature,
	          const size_t sample_size = 10'000)
	    : m(mass), V(voltage), r(cathode_radius), R(anode_radius), T(temprature), VE_cache(PDE_SOL) {
		constexpr double min_energy = 0.01;
		const double max_energy = exp_max_energy(sample_size);
		const double fp_max_energy = fusion_probability(max_energy);
		const auto res =
		    poly_reg<100>([this, fp_max_energy](const double x) { return fusion_probability(x) / fp_max_energy; },
		                  min_energy,
		                  max_energy);
		poly_reg_res = res.second;
		for (size_t i = 0; i < POLY_REG_ORDER; i++) {
			poly_reg_coefs[i] = res.first[i];
		}
	}

	bool find_path(const Vector3d &init_q,
	               const Vector3d &init_p,
	               const double max_dt = 0.25,
	               const double max_fp = 10'000,
	               const bool record_path = false) {
		constexpr int MAX_RETRY = 5;
		constexpr double STEP_REDUCTION = 0.5;
		constexpr double STEP_INCREASE = 1.5;
		constexpr double MIN_DT = 1e-3;

		if (reset(init_q, init_p, record_path)) {
			double t = 0.0;
			double dt = max_dt;
			auto state = std::make_pair(init_q, init_p);

			bool oob = false;
			auto system = std::make_pair([&](auto const &p, auto &dq) { dq = p / m; },
			                             [&](auto const &q, auto &dp) {
				                             if (VE_cache.E_eval(q)) [[likely]] {
					                             dp = VE_cache.E;
				                             } else {
					                             oob = true;
					                             dp.setZero();
				                             }
			                             });
			odeint::symplectic_rkn_sb3a_m4_mclachlan<Vector3d> stepper;

			try {
				while (max_fp < 0 || path_info.fusion_prop * path_info.orbit_cnt < max_fp) {
					oob = false;
					const auto cur_state = state;
					for (int retry_cnt = 0; retry_cnt <= MAX_RETRY; retry_cnt++) {
						stepper.do_step(system, state, t, dt);
						if (state_is_not_valid(state)) [[unlikely]] {
							return false;
						}
						if (!oob) [[likely]] {
							dt = std::min(dt * STEP_INCREASE, max_dt);
							break;
						} else {
							if (retry_cnt == MAX_RETRY) {
								return true;
							} else {
								state = cur_state;
								dt = std::max(dt * STEP_REDUCTION, MIN_DT);
							}
						}
					}
					t += dt;
					observe(state, t, max_dt);
				}
			} catch (const dealii::ExceptionBase &e) {
				std::cerr << "[deal.II] " << e.what() << std::endl;
			} catch (const std::exception &e) {
				std::cerr << "[std] " << e.what() << std::endl;
			} catch (...) {
				std::cerr << "unknown fatal error" << std::endl;
			}
		}
		return false;
	}
};

#endif
