#include <algorithm>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <cmath>
#include <numeric>

namespace {
double find_ks(const std::vector<double> &x, std::function<double(double)> cdf) {
    std::vector<double> sort_x = x;
    std::sort(sort_x.begin(), sort_x.end());
    double max_diff = 0;
    for (size_t i = 0; i < sort_x.size(); ++i) {
        const double diff = std::abs(cdf(sort_x[i]) - static_cast<double>(i + 1) / sort_x.size());
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}
} // namespace

class ExpDist {
  public:
    double lambda;
    double ks;

    ExpDist(const std::vector<double> &x) {
        const double sum = std::accumulate(x.begin(), x.end(), 0.0);
        lambda = x.size() / sum;
        ks = find_ks(x, [this](double val) {
            return cdf(val);
        });
    }

    double pdf(const double x) const {
        return lambda * std::exp(-lambda * x);
    }

    double cdf(const double x) const {
        return 1 - std::exp(-lambda * x);
    }
};

class GammaDist {
  public:
    double alpha;
    double lambda;
    double ks;

    GammaDist(const std::vector<double> &x) {
        const size_t N = x.size();
        const double sum_log = std::accumulate(x.begin(), x.end(), 0.0, [](double acc, double x) {
            return acc + std::log(x);
        });
        const double mean_log = sum_log / N;
        const double mean = std::accumulate(x.begin(), x.end(), 0.0) / N;
        const double log_mean = std::log(mean);

        auto gs = [mean_log, log_mean](const double a) {
            return boost::math::digamma(a) - std::log(a) - mean_log + log_mean;
        };

        auto dgs = [](const double a) {
            return boost::math::trigamma(a) - 1 / a;
        };

        alpha = 1 / (2 * (log_mean - mean_log));
        double d = 1;
        size_t iter = 0;
        while (std::abs(d) > 1e-8 && iter < 100) {
            d = gs(alpha) / dgs(alpha);
            alpha -= d;
            iter++;
        }
        lambda = alpha / mean;

        ks = find_ks(x, [this](double val) {
            return cdf(val);
        });
    }

    double pdf(const double x) const {
        return (std::pow(lambda, alpha) / std::tgamma(alpha)) * std::pow(x, alpha - 1) * std::exp(-lambda * x);
    }

    double cdf(const double x) const {
        return boost::math::gamma_p(alpha, lambda * x);
    }
};
