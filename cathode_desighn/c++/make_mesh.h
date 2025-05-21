#ifndef MAKE_MESH_H
#define MAKE_MESH_H

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <gmsh.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;
using Vector3d = Eigen::Vector3d;
namespace geo = gmsh::model::geo;

// public interface
inline void make_mesh(int app_cnt);

// private
namespace {

void get_data(std::vector<Vector3d> &vertices, std::vector<std::pair<int, int>> &edges, int app_cnt) {
    std::string filename = "../../cathode_data/appratures_" + std::to_string(app_cnt) + ".json";

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    json json_data = json::parse(file);
    for (const auto &json_vertex : json_data["vertices"]) {
        auto v = json_vertex.get<std::vector<double>>();
        vertices.emplace_back(v[0], v[1], v[2]);
    }

    for (const auto &json_edge : json_data["edges"]) {
        auto v1 = json_edge[0].get<std::vector<double>>();
        auto v2 = json_edge[1].get<std::vector<double>>();
        Vector3d e1(v1[0], v1[1], v1[2]);
        Vector3d e2(v2[0], v2[1], v2[2]);
        int idx1, idx2;
        for (size_t i = 0; i < vertices.size(); i++) {
            if ((vertices[i] - e1).norm() < 1e-6) {
                idx1 = i;
            }
            if ((vertices[i] - e2).norm() < 1e-6) {
                idx2 = i;
            }
        }

        edges.emplace_back(std::min(idx1, idx2), std::max(idx1, idx2));
    }

    file.close();
}

std::vector<std::vector<int>> find_adj_list(std::vector<Vector3d> vertices, std::vector<std::pair<int, int>> edges) {
    std::vector<std::vector<int>> adj_list(vertices.size());
    for (const auto &[e1, e2] : edges) {
        adj_list[e1].push_back(e2);
        adj_list[e2].push_back(e1);
    }

    for (size_t i = 0; i < vertices.size(); i++) {
        const Vector3d axis = vertices[i];
        const Vector3d b1 = axis.cross(Vector3d::UnitZ()).normalized();
        const Vector3d b2 = axis.cross(b1).normalized();

        std::vector<double> angles;
        for (const auto adj : adj_list[i]) {
            const Vector3d v = vertices[adj];
            const Vector3d p = v - axis.dot(v) * axis;
            angles.push_back(std::atan2(p.dot(b1), p.dot(b2)));
        }

        std::vector<size_t> indices(adj_list[i].size());
        for (size_t j = 0; j < indices.size(); j++) {
            indices[j] = j;
        }

        std::sort(indices.begin(), indices.end(), [&angles](size_t a, size_t b) { return angles[a] < angles[b]; });

        std::vector<int> sorted_adj(adj_list[i].size());
        for (size_t j = 0; j < adj_list[i].size(); j++) {
            sorted_adj[j] = adj_list[i][indices[j]];
        }
        adj_list[i] = sorted_adj;
    }

    return adj_list;
}

} // namespace

inline void make_mesh(int app_cnt, double anode_radius, double cathode_radius, double wire_radius, int cathode_resolution, int anode_resolution) {
    const double scaled_wire_radius = wire_radius / cathode_radius;
    const double scaled_anode_radius = anode_radius / cathode_radius;
    const double cathode_mesh_size = scaled_wire_radius * (2 * M_PI / static_cast<double>(cathode_resolution));
    const double anode_mesh_size = scaled_anode_radius * (2 * M_PI / static_cast<double>(anode_resolution));

    std::vector<Vector3d> vertices;
    std::vector<std::pair<int, int>> edges;
    get_data(vertices, edges, app_cnt);
    auto adj_list = find_adj_list(vertices, edges);

    gmsh::initialize();
    gmsh::option::setNumber("General.Terminal", 0);
    gmsh::model::add("Fusion_Reactor");

    std::vector<int> geo_vertices;
    for (const Vector3d &v : vertices) {
        geo_vertices.push_back(geo::addPoint(v.x(), v.y(), v.z(), cathode_mesh_size));
    }

    gmsh::finalize();
}

#endif
