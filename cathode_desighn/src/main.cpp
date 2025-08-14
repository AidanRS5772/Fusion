#include "make_mesh.h"
#include "plot.h"
#include "solve_path.h"
#include "solve_pde.h"
#include <Eigen/Dense>
#include <algorithm>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <chrono>
#include <cmath>
#include <deal.II/lac/lapack_support.h>
#include <dlib/optimization.h>
#include <exception>
#include <fstream>
#include <gperftools/profiler.h>
#include <iomanip>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <ostream>
#include <random>
#include <ratio>
#include <stdexcept>
#include <string>

constexpr double cathode_radius = 5;	// [cm]
constexpr double anode_radius = 25;		// [cm]
constexpr double wire_radius = .1;		// [cm]
constexpr double voltage = 1;			// [MV]
constexpr double mD = 2.08690083;		// [MeV][cm/ns]^-2 mass of a deuteron (m = E/c^2)
constexpr double temp = 86.17333262;	// [µeV] room temprature energy
constexpr size_t mc_sample_size = 1000; // number of monte carlo samples

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

	const double den = std::sqrt(K0) + std::sqrt(3 * M_PI / (2 * Ke)) * (K0 + Ke / 3) *
										   std::erf(std::sqrt(3 * K0 / (2 * Ke))) * std::exp(3 * K0 / (2 * Ke));
	return f_int / den;
}

template <size_t N, size_t M>
std::pair<Eigen::Matrix<double, N + 1, 1>, double> poly_reg(const std::function<double(double)> f, const double a,
															const double b) {
	Eigen::Matrix<double, M, 1> Y;
	Eigen::Matrix<double, M, N + 1> V;
	std::vector<double> X(M);
	for (size_t i = 0; i < M; i++) {
		const double x = a + i * (b - a) / (M - 1);
		Y(i) = f(x);
		for (size_t j = 0; j < N + 1; j++) V(i, j) = std::pow(x, j);
	}

	Eigen::Matrix<double, N + 1, 1> coefs = (V.transpose() * V).ldlt().solve(V.transpose() * Y);
	double res = std::sqrt((V * coefs - Y).squaredNorm() / M);
	return {coefs, res};
}

double exp_max_energy() {
	constexpr double gamma = 0.577215664901532;
	return voltage + 1e-6 * temp *
						 (std::log(2) - std::log(M_PI) / 2 + gamma + std::log(mc_sample_size) +
						  std::log(std::log(mc_sample_size)));
}

std::function<double(double)> make_fusion_probability() {
	constexpr size_t order = 2;
	constexpr double min_energy = 0.01;
	const double max_energy = exp_max_energy();
	const double fp_max_energy = fusion_probability(max_energy);
	auto res = poly_reg<order, 100>([fp_max_energy](const double x) { return fusion_probability(x) / fp_max_energy; },
									min_energy, max_energy);
	std::cout << "Polynomial Fit Residue: " << res.second << std::endl;
	auto coefs = res.first;
	return [coefs](const double x) {
		double res = 0;
		for (size_t i = 0; i <= order; i++) {
			res += coefs[i] * std::pow(x, i);
		}
		return res;
	};
}

template <std::size_t N> std::array<std::pair<Vector3d, Vector3d>, N> make_init_states(const unsigned int seed) {
	std::mt19937 gen(seed);
	std::uniform_real_distribution<double> u;
	std::normal_distribution<double> n(0, std::sqrt(mD * temp) * 1e-6);
	constexpr double r3 = cathode_radius * cathode_radius * cathode_radius;
	constexpr double R3 = anode_radius * anode_radius * anode_radius;
	std::array<std::pair<Vector3d, Vector3d>, N> init_states;
	for (size_t i = 0; i < N; ++i) {
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

template <typename T> double find_AD(const std::vector<T> &X, const std::function<double(double)> &cdf) {
	std::vector<T> sorted_X = X;
	std::sort(sorted_X.begin(), sorted_X.end());
	const size_t n = sorted_X.size();
	const double eps = std::numeric_limits<double>::epsilon();

	double sum = 0.0;
	for (size_t i = 0; i < n; ++i) {
		double Fi = std::clamp(cdf(static_cast<double>(sorted_X[i])), eps, 1.0 - eps);
		double Fni = std::clamp(cdf(static_cast<double>(sorted_X[n - 1 - i])), eps, 1.0 - eps);
		sum += (2.0 * (i + 1) - 1.0) * (std::log(Fi) + std::log1p(-Fni)); // log1p for stability
	}
	return -static_cast<double>(n) - sum / static_cast<double>(n);
}

template <typename T> double find_Kurtosis(const std::vector<T> &X) {
	const size_t n = X.size();
	const double m1 = std::accumulate(X.begin(), X.end(), 0.0) / X.size();
	const double m2 = std::accumulate(
		X.begin(), X.end(), 0.0, [m1](const double acc, const T &x) -> double { return acc + (x - m1) * (x - m1); });
	const double m4 = std::accumulate(X.begin(), X.end(), 0.0, [m1](const double acc, const T &x) -> double {
		return acc + (x - m1) * (x - m1) * (x - m1) * (x - m1);
	});
	return (n + 1) * n * (n - 1) * m4 / ((n - 2) * (n - 3) * m2 * m2) -
		   static_cast<double>(3 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3));
}

double exp_pdf(const double lambda, const double x) { return lambda * std::exp(-lambda * x); }

double exp_cdf(const double lambda, const double x) { return 1 - std::exp(-lambda * x); }

struct pareto_params {
	double sigma;
	double xi;
};

pareto_params fit_pareto(const std::vector<double> &X) {
	// p(0) = sigma , p(1) = xi
	const double mean = std::accumulate(X.begin(), X.end(), 0.0) / X.size();
	dlib::matrix<double, 2, 1> params, upper, lower;
	params = {mean, 1};
	upper = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
	lower = {std::numeric_limits<double>::epsilon(), std::numeric_limits<double>::epsilon()};

	auto mle = [X](const dlib::matrix<double, 2, 1> &p) -> double {
		double sum = 0;
		for (const double x : X) {
			const double z = p(0) * p(1) * x;
			if (z < -1) {
				return std::numeric_limits<double>::max();
			} else {
				sum += std::log(1 + z);
			}
		}
		return (1 + 1 / p(1)) * sum - std::log(p(0)) * X.size();
	};
	double _ = dlib::find_min_box_constrained(dlib::bfgs_search_strategy(), dlib::objective_delta_stop_strategy(), mle,
											  dlib::derivative(mle), params, lower, upper);
	return {params(0), params(1)};
}

double pareto_pdf(const pareto_params p, const double x) {
	const double z = p.sigma * p.xi * x;
	if (z < -1) return 0;
	if (p.xi <= 1e-12) return p.sigma * std::exp(-p.sigma * x);
	return p.sigma * std::pow(1 + z, -1 / p.xi - 1);
}

double pareto_cdf(const pareto_params p, const double x) {
	const double z = p.sigma * p.xi * x;
	if (z < -1) return 0;
	if (p.xi < 1e-12) return 1 - std::exp(-p.sigma * x);
	return 1 - std::pow(1 + z, -1 / p.xi);
}

struct peicwise_linear_params {
	double m1;
	double m2;
	double b;
	double t;
	double rmse;
};

peicwise_linear_params fit_peicwise_linear_to_survival(const std::vector<double> &Xraw) {
	// 1) copy, drop nonpositive (can't log), sort
	std::vector<double> X;
	X.reserve(Xraw.size());
	for (double v : Xraw)
		if (v > 0) X.push_back(v);
	std::sort(X.begin(), X.end());
	const size_t n = X.size();
	if (n < 25) throw std::runtime_error("Not enough positive points for piecewise fit");

	// 2) log X and unconditional empirical log-survival
	std::vector<double> log_X(n), log_S(n);
	for (size_t i = 0; i < n; ++i) {
		log_X[i] = std::log(X[i]);
		log_S[i] = std::log(static_cast<double>(n - i) / n); // right-continuous S
	}

	// helpers
	auto tail_alpha_mle = [&](size_t start) -> double {
		// CSN: alpha_hat = 1 + m / sum log(x_i/x_min)
		const double xmin = X[start];
		double sumlog = 0.0;
		for (size_t i = start; i < n; ++i) sumlog += std::log(X[i] / xmin);
		const size_t m = n - start;
		return 1.0 + static_cast<double>(m) / sumlog; // PDF exponent
	};

	auto ks_unconditional = [&](size_t start, double slope) -> double {
		// Compare unconditional survivals: S_theo(x) = S(xmin) * (x/xmin)^{slope}
		const double xmin = X[start];
		const double S0 = static_cast<double>(n - start) / n;
		double maxdiff = 0.0;
		for (size_t i = start; i < n; ++i) {
			const double emp_S = static_cast<double>(n - i) / n;
			const double theo_S = S0 * std::pow(X[i] / xmin, slope);
			const double d = std::fabs(emp_S - theo_S);
			if (d > maxdiff) maxdiff = d;
		}
		return maxdiff;
	};

	auto fit_slope_ls = [&](size_t a, size_t b) -> double {
		// least-squares slope on [a, b) in (log_X, log_S)
		double sumx = 0, sumy = 0, sumxx = 0, sumxy = 0;
		const size_t len = b - a;
		for (size_t i = a; i < b; ++i) {
			const double x = log_X[i], y = log_S[i];
			sumx += x;
			sumy += y;
			sumxx += x * x;
			sumxy += x * y;
		}
		const double denom = len * sumxx - sumx * sumx;
		return (std::fabs(denom) < 1e-14) ? 0.0 : (len * sumxy - sumx * sumy) / denom;
	};

	// 3) search breakpoints
	const size_t min_tail = std::max<size_t>(10, n / 100);
	const size_t min_head = std::max<size_t>(10, n / 100);

	size_t best_idx = min_head;
	double best_ks = std::numeric_limits<double>::infinity();
	double best_m2 = 0.0;

	for (size_t idx = min_head; idx + min_tail < n; ++idx) {
		if (X[idx] <= 0) continue;
		const double alpha = tail_alpha_mle(idx); // PDF exponent
		const double m2 = -(alpha - 1.0);		  // slope of log S
		if (!std::isfinite(m2)) continue;
		const double ks = ks_unconditional(idx, m2);
		if (ks < best_ks) {
			best_ks = ks;
			best_idx = idx;
			best_m2 = m2;
		}
	}

	// 4) head slope and continuity at the knee
	const double m1 = fit_slope_ls(0, best_idx);
	const double log_t = log_X[best_idx];
	const double S0 = static_cast<double>(n - best_idx) / n; // unconditional survival at knee
	const double b = std::log(S0) - m1 * log_t;				 // enforce continuity: m1*log_t + b = log S0

	double rmse = 0.0;
	for (size_t i = 0; i < n; i++) {
		double val;
		if (log_X[i] < log_t) {
			val = m1 * log_X[i] + b;
		} else {
			val = best_m2 * (log_X[i] - log_t) + m1 * log_t + b;
		}
		val -= log_S[i];
		rmse = val * val;
	}
	rmse /= n;

	return {m1, best_m2, b, log_t, std::sqrt(rmse)};
}

std::pair<double, std::vector<double>> excess_survival_and_exceedence(const std::vector<double> &X, double x) {
	std::vector<double> sorted_x = X;
	std::sort(sorted_x.begin(), sorted_x.end());
	const size_t n = sorted_x.size();

	auto it2 = std::upper_bound(sorted_x.begin(), sorted_x.end(), x); // first > x
	size_t idx2 = static_cast<size_t>(it2 - sorted_x.begin());
	size_t idx1 = idx2 - 1;

	double x1 = sorted_x[idx1], x2 = sorted_x[idx2];
	double s1 = static_cast<double>(n - idx1) / n;
	double s2 = static_cast<double>(n - idx2) / n;

	double m = (s2 - s1) / (x2 - x1);

	std::vector<double> exceedence(X.begin() + idx2, X.end());
	return {m * (x - x1) + s1, exceedence};
}

class JSONWriter {
  private:
	std::mutex file_mutex;
	const std::string file_name;
	json data;

  public:
	JSONWriter(const std::string &fname) : file_name(fname) {}

	void write(const size_t app_cnt, const json &res) {
		file_mutex.lock();
		std::string key = "app_cnt_" + std::to_string(app_cnt);
		data[key] = res;

		std::ofstream of(file_name);
		if (of.is_open()) {
			of << data.dump(2);
			of.close();
		} else {
			throw std::runtime_error("Could not open JSON for Monte Carlo Statistics");
			file_mutex.unlock();
		}
		file_mutex.unlock();
	}
};

std::string format_duration(double seconds) {
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(1);

	if (seconds < 60) {
		oss << seconds << " s";
	} else if (seconds < 3600) {
		int mins = static_cast<int>(seconds / 60);
		double secs = seconds - mins * 60;
		oss << mins << " m " << secs << " s";
	} else {
		int hours = static_cast<int>(seconds / 3600);
		int mins = static_cast<int>((seconds - hours * 3600) / 60);
		double secs = seconds - hours * 3600 - mins * 60;
		oss << hours << " h " << mins << " min " << secs << " s";
	}

	return oss.str();
}

void collect_data(const size_t app_cnt, const std::function<double(double)> fprob) {
	auto mesh = MakeMesh(app_cnt, anode_radius, cathode_radius, wire_radius, 4, 24);
	auto pde_sol = SolvePDE(mesh.file_name, voltage, cathode_radius);

	std::random_device rd;
	auto init_states = make_init_states<mc_sample_size>(rd());

	std::vector<double> fusion_probs;
	std::vector<size_t> orbit_cnts;
	fusion_probs.reserve(mc_sample_size);
	size_t zero_orbit_cnt = 0;

	size_t completed = 0;
	const size_t update_interval = std::max(1UL, mc_sample_size / 100);

	const auto start_time = std::chrono::high_resolution_clock::now();
	std::cout << std::endl;
	std::cout << "\n\n";		 // reserve two lines for the HUD
	std::cout << "\033[2A\0337"; // save cursor
	std::cout << std::flush;

	for (auto const &[q, p] : init_states) {
		const auto path_sol = SolvePath(pde_sol, mD, q, p);
		if (path_sol.sucsessful) {
			orbit_cnts.push_back(path_sol.orbit_cnt);
			if (path_sol.orbit_cnt > 0) {
				const double init_energy = path_sol.init_KE + voltage + path_sol.init_PE;
				const double fusion_prob = path_sol.orbit_cnt * fprob(init_energy);
				fusion_probs.push_back(fusion_prob);
			} else {
				zero_orbit_cnt++;
			}
		}

		completed++;
		if (completed % update_interval == 0) {
			const auto cur_time = std::chrono::high_resolution_clock::now();
			const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cur_time - start_time);
			const double avg_ms = static_cast<double>(duration.count()) / (1000 * completed);
			const double percent = 100.0 * completed / mc_sample_size;
			const double eta = (mc_sample_size - completed) * avg_ms / 1000;
			std::cout << "\0338"
					  << "Progress: " << completed << "/" << mc_sample_size << " (" << std::fixed
					  << std::setprecision(1) << percent << "%)\n"
					  << "\x1b[2K\r"
					  << "Average Execution: " << std::fixed << std::setprecision(1) << avg_ms << " ms "
					  << "(ETA: " << format_duration(eta) << ")" << std::flush;
		}
	}

	std::sort(fusion_probs.begin(), fusion_probs.end());
	const size_t fp_size = fusion_probs.size();

	// Orbit mode
	std::unordered_map<size_t, size_t> freq;
	for (auto x : orbit_cnts) ++freq[x];
	const size_t orbit_cnt_mode =
		std::max_element(freq.begin(), freq.end(), [](auto &a, auto &b) { return a.second < b.second; })->first;

	// Orbit avergae
	const double orbit_cnt_avg =
		static_cast<double>(std::accumulate(orbit_cnts.begin(), orbit_cnts.end(), 0.0)) / orbit_cnts.size();

	// Zero Orbit Proportion
	const double zero_orbit_proportion = static_cast<double>(zero_orbit_cnt) / orbit_cnts.size();

	// Fusion Precentiles
	const double fp_median = fusion_probs[fusion_probs.size() / 2];
	const double fp_90 = fusion_probs[static_cast<size_t>(0.9 * fp_size)];
	const double fp_95 = fusion_probs[static_cast<size_t>(0.95 * fp_size)];
	const double fp_99 = fusion_probs[static_cast<size_t>(0.99 * fp_size)];

	// Fusion Moments
	const double fp_avg = std::accumulate(fusion_probs.begin(), fusion_probs.end(), 0.0) / fusion_probs.size();
	const double fp_kurtosis = find_Kurtosis(fusion_probs);

	// Fitting a peicwise linear funstion to the log-log survivability to find where the inflection in the fusion
	// probabilities this results in defining where the exceedence starts
	const auto pl_params = fit_peicwise_linear_to_survival(fusion_probs);
	const double inflection = std::exp(pl_params.t);

	const auto it2 = std::upper_bound(fusion_probs.begin(), fusion_probs.end(), inflection);
	const size_t idx2 = static_cast<size_t>(it2 - fusion_probs.begin());
	const size_t idx1 = idx2 - 1;
	const double x1 = fusion_probs[idx1];
	const double x2 = fusion_probs[idx2];
	const double s1 = static_cast<double>(fp_size - idx1) / fp_size;
	const double s2 = static_cast<double>(fp_size - idx2) / fp_size;
	const double m = (s2 - s1) / (x2 - x1);

	const double inflection_survival = m * (inflection - x1) + s1;

	// make exceedence
	std::vector<double> exceedence;
	for (const double fp : fusion_probs) {
		if (fp > inflection) {
			exceedence.push_back(fp - inflection);
		}
	}

	// fitting an exponential distribution to the fusion probabilities
	const double lambda = 1 / fp_avg;
	const double exp_AD = find_AD(fusion_probs, [lambda](const double x) { return exp_cdf(lambda, x); });
	const double exp_log_tail_diff = std::log(inflection_survival) - std::log(1 - exp_cdf(lambda, inflection));

	// fiting generalized pareto distribution to all the fusion probabilites
	const auto pareto_p = fit_pareto(fusion_probs);
	const double pareto_AD = find_AD(fusion_probs, [pareto_p](const double x) { return pareto_cdf(pareto_p, x); });
	const double pareto_log_tail_diff = std::log(inflection_survival) - std::log(1 - pareto_cdf(pareto_p, inflection));

	// fitting a generalized pareto distribution to just the exceedence
	const auto excess_pareto_p = fit_pareto(exceedence);
	const double excess_pareto_AD =
		find_AD(exceedence, [excess_pareto_p](const double x) { return pareto_cdf(excess_pareto_p, x); });

	std::cout << std::setprecision(6) << std::endl;
	std::cout << "\nOrbit Mode: " << orbit_cnt_mode << std::endl;
	std::cout << "Orbit Average: " << orbit_cnt_avg << std::endl;
	std::cout << "Zero Orbit Proportion: " << zero_orbit_proportion << std::endl;
	std::cout << "\nFusion Average: " << fp_avg << std::endl;
	std::cout << "Fusion Kurtosis: " << fp_kurtosis << std::endl;
	std::cout << "Fusion Median: " << fp_median << std::endl;
	std::cout << "Fusion 90%: " << fp_90 << std::endl;
	std::cout << "Fusion 95%: " << fp_95 << std::endl;
	std::cout << "Fusion 99%: " << fp_99 << std::endl;
	std::cout << "\nPiecewise Log Fit: " << std::endl;
	std::cout << "  m1: " << pl_params.m1 << std::endl;
	std::cout << "  m2: " << pl_params.m2 << std::endl;
	std::cout << "  b: " << pl_params.b << std::endl;
	std::cout << "  t: " << pl_params.t << std::endl;
	std::cout << "  RMSE: " << pl_params.rmse << std::endl;
	std::cout << "\nInflection: " << inflection << std::endl;
	std::cout << "Survival Inflection: " << inflection_survival << std::endl;
	std::cout << "\nExponential Distribution:" << std::endl;
	std::cout << "  lambda: " << lambda << std::endl;
	std::cout << "  AD: " << exp_AD << std::endl;
	std::cout << "  Log Tail Diff: " << exp_log_tail_diff << std::endl;
	std::cout << "\nPareto Distribution:" << std::endl;
	std::cout << "  sigma: " << pareto_p.sigma << std::endl;
	std::cout << "  xi: " << pareto_p.xi << std::endl;
	std::cout << "  AD: " << pareto_AD << std::endl;
	std::cout << "  Log Tail Diff: " << pareto_log_tail_diff << std::endl;
	std::cout << "\nExcess Pareto Distribution:" << std::endl;
	std::cout << "  sigma: " << excess_pareto_p.sigma << std::endl;
	std::cout << "  xi: " << excess_pareto_p.xi << std::endl;
	std::cout << "  AD: " << excess_pareto_AD << std::endl;

	plot_orbits_and_fits(
		fusion_probs, [lambda](const double x) { return exp_pdf(lambda, x); },
		[pareto_p](const double x) { return pareto_pdf(pareto_p, x); });
}

template <size_t T> void Monte_Carlo_Simulation(const size_t max_app_cnt) {
	const auto fprob = make_fusion_probability();
	std::atomic<size_t> app_cnt{6};
	std::vector<std::thread> threads;

	auto worker = [&](size_t thread_id) {
		while (true) {
			size_t cur_app_cnt = app_cnt.fetch_add(2);
			if (cur_app_cnt > max_app_cnt) break;
			try {
				collect_data(cur_app_cnt, fprob);
			} catch (const std::exception &e) {
				std::cerr << "Thread " << thread_id << " failed on app_cnt = " << cur_app_cnt << ": " << e.what()
						  << std::endl;
			}
		}
	};

	for (size_t i = 0; i < T; i++) {
		threads.emplace_back(worker, i);
	}

	for (auto &thread : threads) {
		thread.join();
	}
}

int main() {
	constexpr size_t app_cnt = 6;
	constexpr size_t N = 250;
	constexpr size_t warm_up = 5;
	auto mesh = MakeMesh(app_cnt, anode_radius, cathode_radius, wire_radius);
	auto pde_sol = SolvePDE(mesh.file_name, voltage, cathode_radius);
	auto init_states = make_init_states<N + warm_up>(5772);

	double max_dt = 0.5;
	for (size_t i = 0; i < 6; i++) {
		double sum = 0;
		double sqr_sum = 0;
		size_t cnt = 0;
		std::cout << "\nmax_dt = " << max_dt << std::endl;
		for (const auto &[q, p] : init_states) {
			const auto start = std::chrono::steady_clock::now();

			const auto _ = SolvePath(pde_sol, mD, q, p, max_dt);

			const auto end = std::chrono::steady_clock::now();
			const auto dur = std::chrono::duration<double, std::milli>(end - start).count();

			cnt++;
			if (cnt > warm_up) {
				sum += dur;
				sqr_sum += dur * dur;
				std::cout << "\rProgress: " << cnt - warm_up << "/" << N << std::flush;
			} else {
				std::cout << "\rWarm Up" << std::flush;
			}
		}

		std::cout << "\33[2K\rAverage: " << sum / N << " ms" << std::endl;
		std::cout << "Std Dev: " << std::sqrt((sqr_sum - sum * sum / N) / (N - 1)) << " ms" << std::endl;
		max_dt /= 2;
	}

	return 0;
}
