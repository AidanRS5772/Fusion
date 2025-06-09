#ifndef OCTO_TREE_H
#define OCTO_TREE_H

#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

using Vector3d = Eigen::Vector3d;

class OctoTree {
  private:
    struct AABB {
        Vector3d min, max;

        AABB() {
            min = Vector3d::Constant(std::numeric_limits<double>::max());
            max = Vector3d::Constant(-std::numeric_limits<double>::max());
        }

        AABB(Vector3d min_, Vector3d max_) : min(std::move(min_)), max(std::move(max_)) {}

        bool contains(const Vector3d &p) {
            return (p.array() >= min.array()).all() && (p.array() <= max.array()).all();
        }

        std::array<AABB, 8> split() {
            Vector3d c = (min + max) / 2;
            std::array<AABB, 8> children;
            std::array<bool, 2> bool_arr = {false, true};
            size_t i = 0;
            for (const bool zb : bool_arr) {
                for (const bool yb : bool_arr) {
                    for (const bool xb : bool_arr) {
                        Vector3d new_min(xb ? c.x() : min.x(), yb ? c.y() : min.y(), zb ? c.z() : min.z());
                        Vector3d new_max(xb ? max.x() : c.x(), yb ? max.y() : c.y(), zb ? max.z() : c.z());
                        children[i++] = AABB(new_min, new_max);
                    }
                }
            }
            return children;
        }
    };

    struct Node {
        AABB bounds;
        std::vector<std::array<size_t, 3>> triangles;
        std::array<std::unique_ptr<Node>, 8> children;

        Node() = default;
        Node(const AABB &bounds_, const std::vector<std::array<size_t, 3>> &triangles_)
            : bounds(bounds_), triangles(triangles_) {}

        bool terminal() {
            for (const auto &child : children) {
                if (child) {
                    return false;
                }
            }
            return true;
        }
    };

    std::unique_ptr<Node> root;
    const size_t min_size;
    const size_t max_depth;

    void build(std::unique_ptr<Node> &node, const size_t depth) {
        auto child_bounds = node->bounds.split();
        std::array<std::vector<std::array<size_t, 3>>, 8> child_triangles;
        for (const auto &tri : node->triangles) {
            for (size_t i = 0; i < 8; i++) {
                for (size_t j = 0; j < 3; j++) {
                    if (child_bounds[i].contains(points[tri[j]])) {
                        child_triangles[i].push_back(tri);
                        break;
                    }
                }
            }
        }

        for (size_t i = 0; i < 8; i++) {
            if (!child_triangles[i].empty()) {
                node->children[i] = std::make_unique<Node>(child_bounds[i], child_triangles[i]);
            }
        }
        if (depth < max_depth) {
            for (auto &child : node->children) {
                if (child) {
                    if (child->triangles.size() > min_size) {
                        build(child, depth + 1);
                    }
                }
            }
        }

        node->triangles.clear();
        node->triangles.shrink_to_fit();
    }

    void queryNode(const std::unique_ptr<Node> &node, const Vector3d &point,
                   std::vector<std::array<size_t, 3>> &result) const {
        if (node->terminal()) {
            result.insert(result.end(), node->triangles.begin(), node->triangles.end());
        } else {
            for (const auto &child : node->children) {
                if (child) {
                    if (!child->bounds.contains(point)) {
                        queryNode(child, point, result);
                    }
                }
            }
        }
    }

  public:
    const std::vector<Vector3d> points;

    OctoTree(const std::vector<Vector3d> &points_, const std::vector<std::array<size_t, 3>> &triangles, const double R,
             size_t min_size_ = 16, size_t max_depth_ = 10)
        : min_size(min_size_), max_depth(max_depth_), points(points_) {
        AABB root_bounds(1.1 * Vector3d::Constant(-R), 1.1 * Vector3d::Constant(R));
        root = std::make_unique<Node>(root_bounds, triangles);
        build(root, 0);
    }

    std::optional<std::vector<std::array<size_t, 3>>> query(const Vector3d &point) const {
        if (root->bounds.contains(point)) {
            std::vector<std::array<size_t, 3>> result;
            queryNode(root, point, result);
            return result;
        }
        return std::nullopt;
    }
};

#endif
