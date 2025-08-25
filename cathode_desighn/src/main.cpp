#include "make_mesh.h"
#include "plot.h"
#include "solve_path.h"
#include "solve_pde.h"
#include "stats.h"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <gperftools/profiler.h>
#include <iomanip>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr double cathode_radius = 5;    // [cm]
constexpr double anode_radius = 25;     // [cm]
constexpr double wire_radius = .1;      // [cm]
constexpr double voltage = 1;           // [MV]
constexpr double mD = 2.08690083;       // [MeV][cm/ns]^-2 mass of a deuteron (m = E/c^2)
constexpr double temp = 86.17333262;    // [µeV] room temprature energy
constexpr size_t mc_sample_size = 1000; // number of monte carlo samples

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

void collect_data(const size_t app_cnt) {
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
	std::cout << "\n\n";         // reserve two lines for the HUD
	std::cout << "\033[2A\0337"; // save cursor
	std::cout << std::flush;

	auto path = SolvePath(pde_sol, mD, voltage, cathode_radius, anode_radius, temp, mc_sample_size);
	for (auto const &[q, p] : init_states) {
		if (path.find_path(q, p)) {
			orbit_cnts.push_back(path.path_info.orbit_cnt);
			if (path.path_info.orbit_cnt > 0) {
				fusion_probs.push_back(path.path_info.fusion_prop * path.path_info.orbit_cnt);
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
	std::unordered_map<size_t, size_t> orbit_freq;
	for (auto x : orbit_cnts) ++orbit_freq[x];
	const size_t orbit_cnt_mode = std::max_element(orbit_freq.begin(), orbit_freq.end(), [](auto &a, auto &b) {
		                              return a.second < b.second;
	                              })->first;

	// Orbit avergae
	const double orbit_cnt_avg =
	    static_cast<double>(std::accumulate(orbit_cnts.begin(), orbit_cnts.end(), 0.0)) / orbit_cnts.size();

	// Orbit median
	std::sort(orbit_cnts.begin(), orbit_cnts.end());
	const double orbit_cnt_med = orbit_cnts[orbit_cnts.size() / 2];

	// Zero Orbit Proportion
	const double zero_orbit_proportion = static_cast<double>(zero_orbit_cnt) / orbit_cnts.size();

	// Fusion Precentiles
	const double fp_median = fusion_probs[fusion_probs.size() / 2];
	const double fp_75 = fusion_probs[static_cast<size_t>(0.75 * fp_size)];
	const double fp_90 = fusion_probs[static_cast<size_t>(0.9 * fp_size)];
	const double fp_95 = fusion_probs[static_cast<size_t>(0.95 * fp_size)];
	const double fp_99 = fusion_probs[static_cast<size_t>(0.99 * fp_size)];

	// Fusion Moments
	const double fp_avg = std::accumulate(fusion_probs.begin(), fusion_probs.end(), 0.0) / fusion_probs.size();
	const double fp_var = find_Variance(fusion_probs);
	const double fp_skew = find_Skew(fusion_probs);
	const double fp_kurtosis = find_Kurtosis(fusion_probs);

	const Exponential exp_dist = {1 / fp_avg};
	const double exp_AD = find_AD(fusion_probs, [exp_dist](const double x) -> double { return exp_dist.cdf(x); });

	const auto pareto_dist = fit_pareto(fusion_probs);
	const double pareto_AD =
	    find_AD(fusion_probs, [pareto_dist](const double x) -> double { return pareto_dist.cdf(x); });

	constexpr size_t HAZARD_ORDER = 3;
	const auto hazard_dist = fit_hazard<HAZARD_ORDER>(fusion_probs);
	const double hazard_AD =
	    find_AD(fusion_probs, [hazard_dist](const double x) -> double { return hazard_dist.cdf(x); });

	std::cout << std::setprecision(6) << std::endl;
	std::cout << "\nOrbit Mode: " << orbit_cnt_mode << std::endl;
	std::cout << "Orbit Average: " << orbit_cnt_avg << std::endl;
	std::cout << "Orbit Median: " << orbit_cnt_med << std::endl;
	std::cout << "Zero Orbit Proportion: " << zero_orbit_proportion << std::endl;
	std::cout << "\nFusion Average: " << fp_avg << std::endl;
	std::cout << "Fusion Variance: " << fp_var << std::endl;
	std::cout << "Fusion Skew: " << fp_skew << std::endl;
	std::cout << "Fusion Kurtosis: " << fp_kurtosis << std::endl;
	std::cout << "\nFusion Median: " << fp_median << std::endl;
	std::cout << "Fusion 75%: " << fp_75 << std::endl;
	std::cout << "Fusion 90%: " << fp_90 << std::endl;
	std::cout << "Fusion 95%: " << fp_95 << std::endl;
	std::cout << "Fusion 99%: " << fp_99 << std::endl;
	std::cout << "\nExponential Distribution:" << std::endl;
	std::cout << "  lambda: " << exp_dist.lambda << std::endl;
	std::cout << "  AD: " << exp_AD << std::endl;
	std::cout << "\nPareto Distribution:" << std::endl;
	std::cout << "  lambda: " << pareto_dist.lambda << std::endl;
	std::cout << "  xi: " << pareto_dist.xi << std::endl;
	std::cout << "  AD: " << pareto_AD << std::endl;
	std::cout << "\nHazard Distribution:" << std::endl;
	std::cout << "  U: [";
	for (size_t i = 0; i < HAZARD_ORDER - 1; i++) std::cout << hazard_dist.U[i] << " , ";
	std::cout << hazard_dist.U[HAZARD_ORDER - 1] << "]" << std::endl;
	std::cout << "  V: [";
	for (size_t i = 0; i < HAZARD_ORDER - 2; i++) std::cout << hazard_dist.V[i] << " , ";
	std::cout << hazard_dist.V[HAZARD_ORDER - 2] << "]" << std::endl;
	std::cout << "  alpha: " << hazard_dist.alpha << std::endl;
	std::cout << "  pi: " << hazard_dist.pi() << std::endl;
	std::cout << "  AD: " << hazard_AD << std::endl;

	plot_fp_cdf_and_fits(
	    fusion_probs,
	    [&exp_dist](const double x) { return exp_dist.cdf(x); },
	    [&pareto_dist](const double x) { return pareto_dist.cdf(x); },
	    [&hazard_dist](const double x) { return hazard_dist.cdf(x); });
}

template <size_t T> void Monte_Carlo_Simulation(const size_t max_app_cnt) {
	std::atomic<size_t> app_cnt{6};
	std::vector<std::thread> threads;

	auto worker = [&](size_t thread_id) {
		while (true) {
			size_t cur_app_cnt = app_cnt.fetch_add(2);
			if (cur_app_cnt > max_app_cnt) break;
			try {
				collect_data(cur_app_cnt);
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
	collect_data(42);

	// auto mesh = MakeMesh(6, anode_radius, cathode_radius, wire_radius);
	// auto pde_sol = SolvePDE(mesh.file_name, voltage, cathode_radius);
	// auto path = SolvePath(pde_sol, mD, voltage, cathode_radius, anode_radius, temp);
	// Vector3d q = {0, 0, 17.5}, p = {0, 0, 0};
	//
	// ProfilerStart("path.prof");
	// path.find_path(q, p);
	// ProfilerStop();
	//
	// plot_mesh_path(mesh.hash, path.path_info.states, mD);

	// constexpr size_t app_cnt = 6;
	// constexpr size_t N = 250;
	// constexpr size_t warm_up = 10;
	// auto mesh = MakeMesh(app_cnt, anode_radius, cathode_radius, wire_radius);
	// auto pde_sol = SolvePDE(mesh.file_name, voltage, cathode_radius);
	// auto init_states = make_init_states<N + warm_up>(5772);
	//
	// std::vector<double> times;
	// times.reserve(N);
	// double max_dt = 0.5;
	//
	// auto path = SolvePath(pde_sol, mD, voltage, cathode_radius, anode_radius, temp);
	// for (size_t i = 0; i < 6; i++) {
	// 	size_t cnt = 0;
	// 	std::cout << "\nmax_dt = " << max_dt << std::endl;
	// 	for (const auto &[q, p] : init_states) {
	// 		const auto start = std::chrono::steady_clock::now();
	//
	// 		path.find_path(q, p, max_dt, 50);
	//
	// 		const auto end = std::chrono::steady_clock::now();
	// 		const auto dur = std::chrono::duration<double, std::milli>(end - start).count();
	//
	// 		cnt++;
	// 		if (cnt > warm_up) {
	// 			times.push_back(dur);
	// 			std::cout << "\rProgress: " << cnt - warm_up << "/" << N << std::flush;
	// 		} else {
	// 			std::cout << "\rWarm Up: " << cnt << "/" << warm_up << std::flush;
	// 		}
	// 	}
	//
	// 	const double tot = std::accumulate(times.begin(), times.end(), 0.0, std::plus<double>());
	// 	const double avg = tot / N;
	// 	const double var =
	// 	    std::accumulate(
	// 	        times.begin(), times.end(), 0.0, [avg](const double acc, const double t) -> double { return acc + (t -
	// avg) * (t - avg); }) 	    / (N - 1); 	std::sort(times.begin(), times.end()); 	const double med = times[N / 2];
	//
	// 	std::cout << "\33[2K\rAverage: " << avg << " ms" << std::endl;
	// 	std::cout << "Median: " << med << " ms" << std::endl;
	// 	std::cout << "Std Dev: " << std::sqrt(var) << " ms" << std::endl;
	// 	std::cout << "Total Time: " << tot << " ms" << std::endl;
	//
	// 	max_dt /= 2;
	// 	times.clear();
	// }

	return 0;
}
