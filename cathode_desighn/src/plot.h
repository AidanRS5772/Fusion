#ifndef PLOT_H
#define PLOT_H

#include <Eigen/Dense>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

using Vector3d = Eigen::Vector3d;
using json = nlohmann::json;
using V_func = std::function<std::optional<double>(const Vector3d &)>;
using E_func = std::function<std::optional<Vector3d>(const Vector3d &)>;

inline void plot_mesh(const size_t hash) {
	std::string cmd = "node " + std::string(PROJECT_ROOT) + "/src/plots/plot.js 0 " + std::string(PROJECT_ROOT) + " "
	                  + std::to_string(hash);
	std::system(cmd.c_str());
}

inline void
plot_mesh_path(const size_t hash, const std::vector<std::pair<Vector3d, Vector3d>> &states, const double mass) {
	std::vector<double> X, Y, Z, S;
	X.reserve(states.size());
	Y.reserve(states.size());
	Z.reserve(states.size());
	S.reserve(states.size());
	for (const auto &[q, p] : states) {
		X.push_back(q[0]);
		Y.push_back(q[1]);
		Z.push_back(q[2]);
		S.push_back(p.norm() / (2 * mass));
	}

	json data = {{"X", X}, {"Y", Y}, {"Z", Z}, {"S", S}};
	std::ofstream plot_data(std::string(PROJECT_ROOT) + "/src/plots/plot_data.json");
	plot_data << data.dump();
	plot_data.close();

	std::string cmd = "node " + std::string(PROJECT_ROOT) + "/src/plots/plot.js 1 " + std::string(PROJECT_ROOT) + " "
	                  + std::to_string(hash);
	std::system(cmd.c_str());
}

inline void plot_mesh_electric_feild(const size_t hash, E_func E_feild, const double L, const size_t N = 20) {
	std::vector<double> X, Y, Z, U, V, W, M;
	X.reserve(N * N * N / 2);
	Y.reserve(N * N * N / 2);
	Z.reserve(N * N * N / 2);
	U.reserve(N * N * N / 2);
	V.reserve(N * N * N / 2);
	W.reserve(N * N * N / 2);
	M.reserve(N * N * N / 2);
	auto r = Eigen::VectorXd::LinSpaced(N, -L, L);
	for (size_t i = 0; i < N; i++) {
		for (size_t j = 0; j < N; j++) {
			for (size_t k = 0; k < N; k++) {
				Vector3d R(r[i], r[j], r[k]);
				if (R.norm() <= L) {
					auto E = E_feild(R);
					if (E.has_value()) {
						Vector3d E_vec = E.value();
						X.push_back(R.x());
						Y.push_back(R.y());
						Z.push_back(R.z());
						U.push_back(E_vec.x());
						V.push_back(E_vec.y());
						W.push_back(E_vec.z());
						M.push_back(E_vec.norm());
					}
				}
			}
		}
	}

	json data = {{"X", X}, {"Y", Y}, {"Z", Z}, {"U", U}, {"V", V}, {"W", W}, {"M", M}};
	std::ofstream plot_data(std::string(PROJECT_ROOT) + "/src/plots/plot_data.json");
	plot_data << data.dump();
	plot_data.close();

	std::string cmd = "node " + std::string(PROJECT_ROOT) + "/src/plots/plot.js 2 " + std::string(PROJECT_ROOT) + " "
	                  + std::to_string(hash);
	std::system(cmd.c_str());
}

inline void plot_energy(std::vector<double> &times, std::vector<std::pair<double, double>> &energies) {
	std::vector<double> KE, PE, E;
	KE.reserve(energies.size());
	PE.reserve(energies.size());
	E.reserve(energies.size());
	for (const auto &energy : energies) {
		KE.push_back(energy.first);
		PE.push_back(energy.second);
		E.push_back(energy.first + energy.second);
	}

	json data = {{"KE", KE}, {"PE", PE}, {"E", E}, {"T", times}};
	std::ofstream plot_data(std::string(PROJECT_ROOT) + "/src/plots/plot_data.json");
	plot_data << data.dump();
	plot_data.close();

	std::string cmd = "node " + std::string(PROJECT_ROOT) + "/src/plots/plot.js 3 " + std::string(PROJECT_ROOT);
	std::system(cmd.c_str());
}

inline void plot_orbits(std::vector<double> &fp) {
	json data = {{"FP", fp}};
	std::ofstream plot_data(std::string(PROJECT_ROOT) + "/src/plots/plot_data.json");
	plot_data << data.dump();
	plot_data.close();
	std::string cmd = "node " + std::string(PROJECT_ROOT) + "/src/plots/plot.js 4 " + std::string(PROJECT_ROOT);
	std::system(cmd.c_str());
}

inline void plot_fp_pdf_and_fits(std::vector<double> &fp,
                                 std::function<double(double)> exp_fit,
                                 std::function<double(double)> pareto_fit,
                                 std::function<double(double)> hazard_fit,
                                 const size_t N = 10000) {
	auto X = Eigen::VectorXd::LinSpaced(N, 0, *std::max_element(fp.begin(), fp.end()));
	std::vector<double> exp_pdf, pareto_pdf, hazard_pdf;
	exp_pdf.reserve(N);
	pareto_pdf.reserve(N);
	hazard_pdf.reserve(N);
	for (const double x : X) {
		exp_pdf.push_back(exp_fit(x));
		pareto_pdf.push_back(pareto_fit(x));
		hazard_pdf.push_back(hazard_fit(x));
	}
	json data = {{"FP", fp}, {"X", X}, {"EXP_PDF", exp_pdf}, {"PARETO_PDF", pareto_pdf}, {"HAZARD_PDF", hazard_pdf}};
	std::ofstream plot_data(std::string(PROJECT_ROOT) + "/src/plots/plot_data.json");
	plot_data << data.dump();
	plot_data.close();
	std::string cmd = "node " + std::string(PROJECT_ROOT) + "/src/plots/plot.js 5 " + std::string(PROJECT_ROOT);
	std::system(cmd.c_str());
}

inline void plot_fp_cdf_and_fits(std::vector<double> &fp,
                                 std::function<double(double)> exp_fit,
                                 std::function<double(double)> pareto_fit,
                                 std::function<double(double)> hazard_fit,
                                 const size_t N = 10000) {
	auto X = Eigen::VectorXd::LinSpaced(N, 0, *std::max_element(fp.begin(), fp.end()));
	std::vector<double> exp_cdf, pareto_cdf, hazard_cdf;
	exp_cdf.reserve(N);
	pareto_cdf.reserve(N);
	hazard_cdf.reserve(N);
	for (const double x : X) {
		exp_cdf.push_back(exp_fit(x));
		pareto_cdf.push_back(pareto_fit(x));
		hazard_cdf.push_back(hazard_fit(x));
	}

	std::vector<double> fp_cdf(fp.size());
	for (size_t i = 0; i < fp.size(); i++) {
		fp_cdf[i] = static_cast<double>(i + 1) / fp.size();
	}

	json data = {{"FP_CDF_X", fp},
	             {"FP_CDF_Y", fp_cdf},
	             {"X", X},
	             {"EXP_PDF", exp_cdf},
	             {"PARETO_PDF", pareto_cdf},
	             {"HAZARD_PDF", hazard_cdf}};
	std::ofstream plot_data(std::string(PROJECT_ROOT) + "/src/plots/plot_data.json");
	plot_data << data.dump();
	plot_data.close();
	std::string cmd = "node " + std::string(PROJECT_ROOT) + "/src/plots/plot.js 6 " + std::string(PROJECT_ROOT);
	std::system(cmd.c_str());
}
#endif
