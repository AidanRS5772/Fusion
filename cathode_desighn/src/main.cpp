#include "make_mesh.h"
#include "plot.h"
#include "solve_path.h"
#include "solve_pde.h"
#include "stats.h"
#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <gmsh.h>
#include <gperftools/profiler.h>
#include <iomanip>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

constexpr double cathode_radius = 5;      // [cm]
constexpr double anode_radius = 25;       // [cm]
constexpr double voltage = 1;             // [MV]
constexpr double mD = 2.08690083;         // [MeV][cm/ns]^-2 mass of a deuteron (m = E/c^2)
constexpr double temp = 86.17333262;      // [µeV] room temprature energy
constexpr size_t mc_sample_size = 10'000; // number of monte carlo samples

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

json find_stats(std::vector<double> fusion_probs, std::vector<size_t> orbit_cnts) {
	std::sort(fusion_probs.begin(), fusion_probs.end());
	const size_t fp_size = fusion_probs.size();
	std::sort(orbit_cnts.begin(), orbit_cnts.end());
	const size_t orbit_size = orbit_cnts.size();

	// Orbit mode
	std::unordered_map<size_t, size_t> orbit_freq;
	for (auto x : orbit_cnts) ++orbit_freq[x];
	const size_t orbit_cnt_mode = std::max_element(orbit_freq.begin(), orbit_freq.end(), [](auto &a, auto &b) {
		                              return a.second < b.second;
	                              })->first;

	// Orbit avergae
	const double orbit_cnt_avg =
	    static_cast<double>(std::accumulate(orbit_cnts.begin(), orbit_cnts.end(), 0.0)) / orbit_size;

	// Orbit median
	const size_t orbit_cnt_med = orbit_cnts[orbit_size / 2];

	// Orbit max
	const size_t orbit_cnt_max = orbit_cnts.back();

	// Zero Orbit Proportion
	const double zero_orbit_proportion = static_cast<double>(orbit_freq[0]) / orbit_cnts.size();

	// Fusion Precentiles
	const double fp_median = fusion_probs[fusion_probs.size() / 2];
	const double fp_75 = fusion_probs[static_cast<size_t>(0.75 * fp_size)];
	const double fp_90 = fusion_probs[static_cast<size_t>(0.9 * fp_size)];
	const double fp_95 = fusion_probs[static_cast<size_t>(0.95 * fp_size)];
	const double fp_99 = fusion_probs[static_cast<size_t>(0.99 * fp_size)];

	// Fusion Max
	const double fp_max = fusion_probs.back();

	// Fusion Moments
	const double fp_avg = std::accumulate(fusion_probs.begin(), fusion_probs.end(), 0.0) / fusion_probs.size();
	const double fp_var = find_Variance(fusion_probs);
	const double fp_skew = find_Skew(fusion_probs);
	const double fp_kurtosis = find_Kurtosis(fusion_probs);

	const Exponential exp_dist = {1 / fp_avg};
	const double exp_AD = find_AD(fusion_probs, [exp_dist](const double x) -> double { return exp_dist.cdf(x); });

	const Generalized_Pareto pareto_dist = fit_pareto(fusion_probs);
	const double pareto_AD =
	    find_AD(fusion_probs, [pareto_dist](const double x) -> double { return pareto_dist.cdf(x); });

	constexpr size_t HAZARD_ORDER = 3;
	const auto hazard_dist = fit_hazard<HAZARD_ORDER>(fusion_probs);
	const double hazard_AD =
	    find_AD(fusion_probs, [hazard_dist](const double x) -> double { return hazard_dist.cdf(x); });

	json res = {
	    {"orbit",
	     {{"mode", orbit_cnt_mode},
	      {"average", orbit_cnt_avg},
	      {"median", orbit_cnt_med},
	      {"orbit_max", orbit_cnt_max},
	      {"zero_proportion", zero_orbit_proportion}}},
	    {"fusion",
	     {{"average", fp_avg},
	      {"variance", fp_var},
	      {"skew", fp_skew},
	      {"kurtosis", fp_kurtosis},
	      {"fusion_max", fp_max},
	      {"percentiles", {{"median", fp_median}, {"75", fp_75}, {"90", fp_90}, {"95", fp_95}, {"99", fp_99}}}}},
	    {"distributions",
	     {{"exponential", {{"lambda", exp_dist.lambda}, {"AD", exp_AD}}},
	      {"pareto", {{"lambda", pareto_dist.lambda}, {"xi", pareto_dist.xi}, {"AD", pareto_AD}}},
	      {"hazard",
	       {{"U", hazard_dist.U},
	        {"V", hazard_dist.V},
	        {"alpha", hazard_dist.alpha},
	        {"pi", hazard_dist.pi()},
	        {"AD", hazard_AD}}}}}};

	return res;
}

class JSONWriter {
  private:
	std::mutex file_mutex;
	const std::string file_name;

	json read() {
		std::ifstream in(file_name);
		if (!in.is_open()) throw std::runtime_error("JSONWriter: file missing for read: " + file_name);

		try {
			json j;
			in >> j;
			if (!j.is_object()) throw std::runtime_error("JSONWriter: json is malformed: " + file_name);
			return j;
		} catch (const std::exception &e) {
			throw std::runtime_error("JSONWriter: json failed to parse: " + file_name + ": " + e.what());
		}
	}

  public:
	explicit JSONWriter(const std::string &fname) : file_name(fname) {
		namespace fs = std::filesystem;
		std::error_code ec;

		if (!fs::exists(file_name, ec)) throw std::runtime_error("JSONWriter: file does not exist: " + file_name);
		if (!fs::is_regular_file(file_name, ec))
			throw std::runtime_error("JSONWriter: path is not a regular file: " + file_name);

		const auto sz = fs::file_size(file_name, ec);
		if (ec) throw std::runtime_error("JSONWriter: file_size failed for " + file_name + ": " + ec.message());

		if (sz == 0) {
			std::ofstream out(file_name, std::ios::out | std::ios::trunc);
			if (!out) throw std::runtime_error("JSONWriter: cannot open for writing: " + file_name);

			out << "{}\n";
			out.flush();
			if (!out) throw std::runtime_error("JSONWriter: failed to seed empty JSON: " + file_name);
		}
	}

	void write(const size_t app_cnt, const json &res) {
		std::lock_guard<std::mutex> lk(file_mutex);

		json doc = read();
		doc["app_cnt_" + std::to_string(app_cnt)] = res;

		std::ofstream out(file_name, std::ios::trunc);
		if (!out.is_open()) throw std::runtime_error("JSONWriter: could not open file for write: " + file_name);
		out << doc.dump(4) << '\n';
	}
};

template <size_t T> class ProgressTracker {
	std::mutex print_mutex;
	std::array<std::chrono::time_point<std::chrono::steady_clock>, T> starts{};
	std::array<size_t, T> app_cnts{};
	std::array<size_t, T> cnts{};

	static constexpr size_t LINE_CNT = 3;

	static void save_origin() { std::cout << "\x1b" << "7"; }
	static void restore_origin() { std::cout << "\x1b" << "8"; }
	static void move_down(size_t n) {
		if (n) std::cout << "\x1b[" << n << "B";
	}
	static void move_up(size_t n) {
		if (n) std::cout << "\x1b[" << n << "A";
	}
	static void clear_line() { std::cout << "\x1b[2K\r"; }

	static void clear_block(size_t lines) {
		for (size_t i = 0; i < lines; ++i) {
			clear_line();
			if (i + 1 < lines) move_down(1);
		}
		move_up(lines - 1);
	}

	std::string format_duration(double seconds) const {
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
			oss << hours << " h " << mins << " m " << secs << " s";
		}
		return oss.str();
	}

  public:
	const size_t tick_rate;

	ProgressTracker(size_t tick_rate_) : tick_rate(tick_rate_) {
		std::lock_guard<std::mutex> lk(print_mutex);
		std::cout << "\x1b[?25l";
		for (size_t i = 0; i < T * LINE_CNT; ++i) std::cout << "\n";
		move_up(T * LINE_CNT);
		save_origin();
		std::cout << std::flush;
	}

	~ProgressTracker() {
		restore_origin();
		move_down(T * LINE_CNT);
		std::cout << "\x1b[?25h" << std::endl; // show cursor again
	}

	void init(size_t thread_id, size_t app_cnt) {
		starts[thread_id] = std::chrono::steady_clock::now();
		cnts[thread_id] = 0;
		app_cnts[thread_id] = app_cnt;
		update(thread_id, true);
	}

	void update(size_t thread_id, bool force = false) {
		cnts[thread_id] += 1;
		if (!force && cnts[thread_id] % tick_rate != 0) return;

		const auto cur_time = std::chrono::steady_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cur_time - starts[thread_id]);
		const double avg_ms = static_cast<double>(duration.count()) / (1000 * cnts[thread_id]);
		const double percent = 100.0 * cnts[thread_id] / mc_sample_size;
		const double eta = (mc_sample_size - cnts[thread_id]) * avg_ms / 1000;

		std::ostringstream os;
		os << "App Cnt: " << app_cnts[thread_id] << "\n"
		   << "Progress: " << cnts[thread_id] << "/" << mc_sample_size << " (" << std::fixed << std::setprecision(2)
		   << percent << "%)\n"
		   << "Avg Exec: " << std::fixed << std::setprecision(2) << avg_ms << " ms (ETA: " << format_duration(eta)
		   << ")";

		std::lock_guard<std::mutex> lk(print_mutex);
		restore_origin();
		move_down(thread_id * LINE_CNT);
		clear_block(LINE_CNT);
		std::cout << os.str() << std::flush;
		restore_origin();
	}
};

template <size_t T>
json collect_data(const size_t app_cnt, std::shared_ptr<ProgressTracker<T>> progress, size_t thread_id) {
	auto pde_sol = SolvePDE("/meshes/app_" + std::to_string(app_cnt) + ".msh", voltage, cathode_radius);

	// std::random_device rd;
	auto init_states = make_init_states<mc_sample_size>(31415926);

	std::vector<double> fusion_probs;
	std::vector<size_t> orbit_cnts;
	fusion_probs.reserve(mc_sample_size);
	orbit_cnts.reserve(mc_sample_size);

	auto path = SolvePath(pde_sol, mD, voltage, cathode_radius, anode_radius, temp, mc_sample_size);
	progress->init(thread_id, app_cnt);
	for (auto const &[q, p] : init_states) {
		if (path.find_path(q, p)) {
			orbit_cnts.push_back(path.path_info.orbit_cnt);
			if (path.path_info.orbit_cnt > 0) {
				fusion_probs.push_back(path.path_info.fusion_prop * path.path_info.orbit_cnt);
			}
		}
		progress->update(thread_id);
	}

	return find_stats(fusion_probs, orbit_cnts);
}

template <size_t T> void Monte_Carlo_Simulation(const std::vector<size_t> &app_cnts) {
	std::atomic<size_t> app_cnt_idx{0};
	std::vector<std::thread> threads;
	JSONWriter json_writer(std::string(PROJECT_ROOT) + "/MC_data.json");
	std::shared_ptr<ProgressTracker<T>> progress_tracker =
	    std::make_shared<ProgressTracker<T>>(std::max<size_t>(1, mc_sample_size / 100));

	auto worker = [&](size_t thread_id) {
		while (true) {
			const size_t cur_app_cnt_idx = app_cnt_idx.fetch_add(1);
			if (cur_app_cnt_idx >= app_cnts.size()) break;
			const size_t cur_app_cnt = app_cnts[cur_app_cnt_idx];
			try {
				json stats = collect_data(cur_app_cnt, progress_tracker, thread_id);
				json_writer.write(cur_app_cnt, stats);
			} catch (const std::exception &e) {
				std::cerr << "Thread " << thread_id << " failed on app_cnt = " << cur_app_cnt << ": " << e.what()
				          << std::endl;
			}
		}
	};

	for (size_t i = 0; i < T; i++) threads.emplace_back(worker, i);
	for (auto &thread : threads) thread.join();
}

void cathode_visualize(size_t app_cnt) {
	try {
		gmsh::initialize();
		gmsh::option::setNumber("General.Terminal", 1);

		gmsh::open(std::string(PROJECT_ROOT) + "/meshes/app_" + std::to_string(app_cnt) + ".msh");

		int dim = 2;
		int physTag = 1;

		std::vector<int> cathode_tags;
		gmsh::model::getEntitiesForPhysicalGroup(dim, physTag, cathode_tags);

		std::vector<std::pair<int, int>> all;
		gmsh::model::getEntities(all);
		for (auto &e : all) gmsh::model::setVisibility({e}, false);

		for (auto &t : cathode_tags) gmsh::model::setVisibility({std::make_pair(dim, t)}, true);

		gmsh::fltk::initialize();
		gmsh::fltk::run();

		gmsh::finalize();
	} catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << "\n";
	}
}

int main() {
	Monte_Carlo_Simulation<1>({160});
	return 0;
}
