#ifndef STATS_H
#define STATS_H

#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <dlib/optimization.h>
#include <functional>
#include <limits>
#include <vector>

double find_AD(const std::vector<double> &X, const std::function<double(double)> &cdf) {
	std::vector<double> sorted_X = X;
	std::sort(sorted_X.begin(), sorted_X.end());
	const size_t n = sorted_X.size();
	const double eps = std::numeric_limits<double>::epsilon();

	double sum = 0.0;
	for (size_t i = 0; i < n; ++i) {
		double Fi = std::clamp(cdf(static_cast<double>(sorted_X[i])), eps, 1.0 - eps);
		double Fni = std::clamp(cdf(static_cast<double>(sorted_X[n - 1 - i])), eps, 1.0 - eps);
		sum += (2.0 * (i + 1) - 1.0) * (std::log(Fi) + std::log1p(-Fni));
	}
	return -static_cast<double>(n) - sum / static_cast<double>(n);
}

double find_Variance(const std::vector<double> &X) {
	const size_t n = X.size();
	const double m1 = std::accumulate(X.begin(), X.end(), 0.0) / n;
	return std::accumulate(X.begin(),
	                       X.end(),
	                       0.0,
	                       [m1](const double acc, const double x) -> double { return acc + (x - m1) * (x - m1); })
	       / (n - 1);
}

double find_Skew(const std::vector<double> &X) {
	const size_t n = X.size();
	const double m1 = std::accumulate(X.begin(), X.end(), 0.0) / X.size();
	const double M2 = std::accumulate(X.begin(), X.end(), 0.0, [m1](const double acc, const double x) -> double {
		return acc + (x - m1) * (x - m1);
	});
	const double M3 = std::accumulate(X.begin(), X.end(), 0.0, [m1](const double acc, const double x) -> double {
		return acc + (x - m1) * (x - m1) * (x - m1);
	});
	return (n * std::sqrt(n - 1) / (n - 2)) * M3 / std::sqrt(M2 * M2 * M2);
}

double find_Kurtosis(const std::vector<double> &X) {
	const size_t n = X.size();
	const double m1 = std::accumulate(X.begin(), X.end(), 0.0) / X.size();
	const double M2 = std::accumulate(X.begin(), X.end(), 0.0, [m1](const double acc, const double x) -> double {
		return acc + (x - m1) * (x - m1);
	});
	const double M4 = std::accumulate(X.begin(), X.end(), 0.0, [m1](const double acc, const double x) -> double {
		return acc + (x - m1) * (x - m1) * (x - m1) * (x - m1);
	});
	return (n + 1) * n * (n - 1) * M4 / ((n - 2) * (n - 3) * M2 * M2)
	       - static_cast<double>(3 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3));
}

struct Exponential {
	double lambda;

	double pdf(const double x) const { return lambda * std::exp(-lambda * x); }

	double cdf(const double x) const { return 1 - std::exp(-lambda * x); }
};

struct Generalized_Pareto {
	const double lambda;
	const double xi;

	static constexpr double XI_LIM = 1e-8;

	Generalized_Pareto(dlib::matrix<double, 2, 1> p) : lambda(p(0)), xi(p(1)) {}

	double pdf(const double x) const {
		const double z = lambda * xi * x;
		if (xi <= XI_LIM) return lambda * std::exp(-lambda * x);
		if (z < -1) return 0;
		return lambda * std::pow(1 + z, -1 / xi - 1);
	}

	double cdf(const double x) const {
		const double z = lambda * xi * x;
		if (xi < XI_LIM) return 1 - std::exp(-lambda * x);
		if (z < -1) return 0;
		return 1 - std::pow(1 + z, -1 / xi);
	}

	double l(const std::vector<double> &X) const {
		if (xi < XI_LIM) {
			const double sum = std::accumulate(X.begin(), X.end(), 0.0);
			return lambda * sum - X.size() * std::log(lambda);
		} else {
			const double log_sum = std::accumulate(X.begin(), X.end(), 0.0, [this](const double acc, const double x) {
				return acc + std::log(1 + xi * lambda * x);
			});
			return (1 + 1 / xi) * log_sum - std::log(lambda) * X.size();
		}
	}
};

Generalized_Pareto fit_pareto(const std::vector<double> &X) {
	// p(0) = lambda , p(1) = xi
	const double mean = std::accumulate(X.begin(), X.end(), 0.0) / X.size();
	dlib::matrix<double, 2, 1> params, upper, lower;
	params = 1 / mean, 1;
	upper = 1, 5;
	lower = 0, 0;

	auto mle = [&X](const dlib::matrix<double, 2, 1> &p) -> double { return Generalized_Pareto(p).l(X); };
	double _ = dlib::find_min_box_constrained(dlib::bfgs_search_strategy(),
	                                          dlib::objective_delta_stop_strategy(),
	                                          mle,
	                                          dlib::derivative(mle),
	                                          params,
	                                          lower,
	                                          upper);
	return {params};
}

template <size_t N> struct Hazard {
	std::array<double, N> U{};
	std::array<double, N - 1> V{};
	double alpha;

	Hazard(dlib::matrix<double, 2 * N, 1> p) {
		for (size_t i = 0; i < N; i++) U[i] = p(i);
		for (size_t i = 0; i < N - 1; i++) V[i] = p(N + i);
		alpha = p(2 * N - 1);
	}

	double h(const double x) const {
		double p = 1;
		double u = 0;
		for (size_t i = 0; i < N; i++) {
			u += U[i] * p;
			p *= x;
		}

		p = x;
		double v = 0;
		for (size_t i = 0; i < N - 1; i++) {
			v += V[i] * p;
			p *= x;
		}
		v += p;

		return std::pow(1 + x, 1 + alpha) * u * u / (1 + v * v);
	}

	double H(const double x) const {
		return boost::math::quadrature::gauss_kronrod<double, 15>::integrate(
		    [this](const double u) { return h(u); }, 0, x);
	}

	double pdf(const double x) const { return h(x) * std::exp(-H(x)); }

	double cdf(const double x) const { return 1 - std::exp(-H(x)); }

	double l(const std::vector<double> &X) const {
		double sum = 0;
		for (const double x : X) sum += H(x) - std::log(std::max(h(x), std::numeric_limits<double>::epsilon()));
		return sum;
	}

	double pi() const {
		if (alpha >= 0.0) return 0.0;

		const double T = 1e3;
		const double H0T = boost::math::quadrature::gauss_kronrod<double, 63>::integrate(
		    [this](double x) { return this->h(x); }, 0.0, T);

		const double tail = (h(T) * T) / (-alpha);

		const double Hinf = H0T + tail;
		const double Hc = std::min(Hinf, 800.0);
		return std::exp(-Hc);
	}
};

template <size_t N> Hazard<N> fit_hazard(const std::vector<double> &X) {
	dlib::matrix<double, 2 * N, 1> params, upper, lower;
	params = 1;
	upper = 1e6;
	upper(2 * N - 1) = 5;
	lower = -1e6;
	lower(2 * N - 1) = -5;

	auto mle = [&X](const dlib::matrix<double, 2 * N, 1> &p) -> double { return Hazard<N>(p).l(X); };
	double _ = dlib::find_min_box_constrained(dlib::bfgs_search_strategy(),
	                                          dlib::objective_delta_stop_strategy(),
	                                          mle,
	                                          dlib::derivative(mle),
	                                          params,
	                                          lower,
	                                          upper);
	return {params};
}

#endif
