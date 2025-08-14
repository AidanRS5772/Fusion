#ifndef MAKE_MESH_H
#define MAKE_MESH_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <gmsh.h>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;
using Vector3d = Eigen::Vector3d;
namespace geo = gmsh::model::geo;

class MakeMesh {
  public:
	std::string file_name;
	size_t hash;

	MakeMesh(const int app_cnt_, const double anode_radius_, const double cathode_radius_, const double wire_radius_,
			 const int cathode_resolution = 4, const int anode_resolution = 24, const bool print = false)
		: app_cnt(app_cnt_), anode_radius(anode_radius_), cathode_radius(cathode_radius_), wire_radius(wire_radius_) {

		scaled_wire_radius = wire_radius / cathode_radius;
		scaled_anode_radius = anode_radius / cathode_radius;
		cathode_mesh_size = scaled_wire_radius * (2 * M_PI / static_cast<double>(cathode_resolution));
		anode_mesh_size = scaled_anode_radius * (2 * M_PI / static_cast<double>(anode_resolution));

		gmsh::initialize();
		gmsh::option::setNumber("General.Terminal", 0);
		gmsh::model::add("Fusion_Reactor");

		if (check_cache(anode_resolution, cathode_resolution)) {
			if (print) std::cout << "\nMesh Found in Cache" << std::endl;
			gmsh::open(std::string(PROJECT_ROOT) + "/mesh_cache/" + file_name);
		} else {
			std::cout << "\nBegining mesh contruction..." << std::endl;

			auto start = std::chrono::high_resolution_clock::now();

			get_data();
			find_adj_list();

			for (const Vector3d &v : vertices) {
				geo_vertices.push_back(geo::addPoint(v.x(), v.y(), v.z(), cathode_mesh_size));
			}

			find_intersection_ellipses();
			find_vertex_curves();
			find_edge_curves();
			geo::synchronize();

			make_surfaces();
			int cathode = geo::addSurfaceLoop(cathode_surfaces);
			auto [anode, anode_surfaces] = make_anode();
			geo::synchronize();

			int chamber = geo::addVolume({anode, -cathode});
			geo::synchronize();

			int cathode_physical = gmsh::model::addPhysicalGroup(2, cathode_surfaces, 1);
			gmsh::model::setPhysicalName(2, cathode_physical, "cathode");

			int anode_physical = gmsh::model::addPhysicalGroup(2, anode_surfaces, 2);
			gmsh::model::setPhysicalName(2, anode_physical, "anode");

			int chamber_physical = gmsh::model::addPhysicalGroup(3, {chamber}, 3);
			gmsh::model::setPhysicalName(3, chamber_physical, "chamber");

			apply_mesh_settings(cathode_surfaces, app_cnt, scaled_wire_radius, cathode_mesh_size, anode_mesh_size);

			if (print) std::cout << "Starting mesh generation..." << std::flush;
			gmsh::model::mesh::generate(3);
			if (print) std::cout << " finished generation." << std::endl;

			for (int i = 0; i < 3; i++) {
				if (print) std::cout << "Optimization Pass " << i + 1 << std::endl;
				gmsh::model::mesh::optimize("Relocate3D");
				gmsh::model::mesh::optimize("Netgen");
				gmsh::model::mesh::optimize("Laplace2D");
			}

			gmsh::write(std::string(PROJECT_ROOT) + "/mesh_cache/" + file_name);

			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

			if (print)
				std::cout << "Mesh construction finished in " << static_cast<double>(duration.count()) / 1000 << " s"
						  << std::endl;
			if (print) std::cout << "Mesh Saved too: mesh_cache/" << file_name << std::endl;
		}

		load_cathode_data();
		if (print) print_mesh_statistics();

		gmsh::finalize();
	}

  private:
	struct Ellipse {
		Vector3d start;
		int geo_start;
		Vector3d end;
		int geo_end;
		Vector3d center;
		int geo_center;
		Vector3d major;
		int geo_major;
		Vector3d minor;
		int geo_ellipse;

		Ellipse()
			: start(Vector3d::Zero()), geo_start(-1), end(Vector3d::Zero()), geo_end(-1), center(Vector3d::Zero()),
			  geo_center(-1), major(Vector3d::Zero()), geo_major(-1), minor(Vector3d::Zero()), geo_ellipse(-1) {}

		Ellipse(const double mesh_size, Vector3d start_, Vector3d end_, Vector3d center_, Vector3d major_,
				Vector3d minor_, std::optional<int> geo_start_ = std::nullopt,
				std::optional<int> geo_end_ = std::nullopt, std::optional<int> geo_center_ = std::nullopt,
				std::optional<int> geo_major_ = std::nullopt)
			: start(std::move(start_)), end(std::move(end_)), center(std::move(center_)), major(std::move(major_)),
			  minor(std::move(minor_)) {
			if (geo_start_.has_value()) {
				geo_start = geo_start_.value();
			} else {
				geo_start = geo::addPoint(start.x(), start.y(), start.z(), mesh_size);
			}

			if (geo_end_.has_value()) {
				geo_end = geo_end_.value();
			} else {
				geo_end = geo::addPoint(end.x(), end.y(), end.z(), mesh_size);
			}

			if (geo_center_.has_value()) {
				geo_center = geo_center_.value();
			} else {
				geo_center = geo::addPoint(center.x(), center.y(), center.z());
			}

			if (geo_major_.has_value()) {
				geo_major = geo_major_.value();
			} else {
				Vector3d geo_major_vec = center + major * mesh_size;
				geo_major = geo::addPoint(geo_major_vec.x(), geo_major_vec.y(), geo_major_vec.z(), mesh_size);
			}

			Vector3d n = major.cross(minor).normalized();
			try {
				geo_ellipse = geo::addEllipseArc(geo_start, geo_center, geo_major, geo_end, -1, n.x(), n.y(), n.z());
			} catch (...) {
				std::cerr << "Failed to make Ellipse" << std::endl;
				geo_ellipse = -1;
			}
		}
	};

	struct VertexCurves {
		Vector3d vertex;
		int geo_vertex;
		std::array<Vector3d, 5> points;
		std::array<int, 5> geo_points;
		std::array<int, 5> curves;

		VertexCurves() = default;

		VertexCurves(const double mesh_size, const double radius, const Vector3d &axis, const Ellipse &inter1,
					 const Ellipse &inter2, Vector3d vertex_, int geo_vertex_)
			: vertex(std::move(vertex_)), geo_vertex(geo_vertex_) {
			points[4] = inter2.start;
			geo_points[4] = inter2.geo_start;
			curves[0] = geo::addCircleArc(inter2.geo_start, geo_vertex, inter1.geo_start);

			points[0] = inter1.start;
			geo_points[0] = inter1.geo_start;
			curves[1] = inter1.geo_ellipse;

			const Vector3d p1 = inter1.end;
			const Vector3d p2 = inter2.end;
			const double t1 = (p1 - vertex).dot(axis);
			const double t2 = (p2 - vertex).dot(axis);

			const Vector3d center = vertex + ((t1 + t2) / 2) * axis;
			const int geo_c = geo::addPoint(center.x(), center.y(), center.z(), mesh_size);

			const Vector3d normal = (p1 - center).cross(p2 - center).normalized();
			const Vector3d minor = center.cross(axis).normalized();
			const Vector3d major = minor.cross(axis).normalized();

			const Vector3d major_vec = center + major * mesh_size;
			const int geo_M = geo::addPoint(major_vec.x(), major_vec.y(), major_vec.z(), mesh_size);

			const Vector3d mid =
				((p1 - center).normalized() + (p2 - center).normalized()).normalized() * radius + center;
			const int geo_mid = geo::addPoint(mid.x(), mid.y(), mid.z(), mesh_size);

			points[1] = inter1.end;
			geo_points[1] = inter1.geo_end;
			curves[2] =
				geo::addEllipseArc(inter1.geo_end, geo_c, geo_M, geo_mid, -1, normal.x(), normal.y(), normal.z());

			points[2] = mid;
			geo_points[2] = geo_mid;
			curves[3] =
				geo::addEllipseArc(geo_mid, geo_c, geo_M, inter2.geo_end, -1, normal.x(), normal.y(), normal.z());

			points[3] = inter2.end;
			geo_points[3] = inter2.geo_end;
			curves[4] = -inter2.geo_ellipse;

			try {
				geo::addCurveLoop({curves[0], curves[1], curves[2], curves[3], curves[4]});
			} catch (...) {
				std::cerr << "Vertex Curves not orientated properly" << std::endl;
			}
		}
	};

	struct EdgeCurves {
		Vector3d vertex;
		int geo_vertex;
		Vector3d axis;
		std::array<int, 10> loops;

		EdgeCurves(const double mesh_size, const double radius, const VertexCurves &v1, const VertexCurves &v2) {
			vertex = (v1.vertex + v2.vertex) / 2;
			geo_vertex = geo::addPoint(vertex.x(), vertex.y(), vertex.z(), mesh_size);
			axis = (v2.vertex - v1.vertex).normalized();

			std::array<std::pair<int, int>, 5> cons;
			std::array<int, 5> mids;
			for (size_t i = 0; i < 5; i++) {
				auto [c1, c2, mid] = find_line_or_ellipse(mesh_size, radius, v1.points[i], v1.geo_points[i],
														  v2.points[4 - i], v2.geo_points[4 - i]);
				cons[i] = std::pair<int, int>(c1, c2);
				mids[i] = mid;
			}

			std::array<int, 5> mid_circs;
			for (size_t i = 0; i < 5; i++) {
				mid_circs[i] = geo::addCircleArc(mids[(i + 4) % 5], geo_vertex, mids[i]);
			}

			for (size_t i = 0; i < 5; i++) {
				loops[2 * i] =
					geo::addCurveLoop({v1.curves[i], cons[i].first, -mid_circs[i], -cons[(i + 4) % 5].first});
				loops[2 * i + 1] = geo::addCurveLoop(
					{mid_circs[i], cons[i].second, v2.curves[(5 - i) % 5], -cons[(i + 4) % 5].second});
			}
		}

	  private:
		std::tuple<int, int, int> find_line_or_ellipse(const double mesh_size, const double radius, Vector3d p1,
													   int geo_p1, Vector3d p2, int geo_p2,
													   const double thresh = 1e-6) {
			const Vector3d l = (p2 - p1).normalized();
			if (l.cross(axis).norm() < thresh) {
				const Vector3d p = (p1 + p2) / 2;
				const Vector3d mid = p + ((vertex - p).dot(axis) / l.dot(axis)) * l;
				assert(std::abs((mid - vertex).norm() - radius) < thresh &&
					   std::abs((mid - vertex).dot(axis)) < thresh && "mid is not a radius away from the vertex");

				const int geo_mid = geo::addPoint(mid.x(), mid.y(), mid.z(), mesh_size);
				const int line1 = geo::addLine(geo_p1, geo_mid);
				const int line2 = geo::addLine(geo_mid, geo_p2);
				return {line1, line2, geo_mid};
			} else {
				const double t1 = (p1 - vertex).dot(axis);
				const double t2 = (p2 - vertex).dot(axis);
				const double tm = (t1 + t2) / 2;

				const Vector3d center = vertex + tm * axis;
				const int geo_c = geo::addPoint(center.x(), center.y(), center.z(), mesh_size);

				const Vector3d normal = (p1 - center).cross(p2 - center).normalized();
				const Vector3d minor = normal.cross(axis).normalized() * radius;

				const Vector3d major = minor.cross(normal).normalized() * (radius / normal.dot(axis));
				const Vector3d vec_M = center + major.normalized() * mesh_size;
				const int geo_M = geo::addPoint(vec_M.x(), vec_M.y(), vec_M.z(), mesh_size);

				const double k = -tm / major.dot(axis);
				Vector3d mid = center + major * k - minor * std::sqrt(1 - k * k);
				assert(std::abs((mid - vertex).norm() - radius) < thresh &&
					   std::abs((mid - vertex).dot(axis)) < thresh && "mid is not a radius away from the vertex");
				const int geo_mid = geo::addPoint(mid.x(), mid.y(), mid.z(), mesh_size);

				const int ell1 =
					geo::addEllipseArc(geo_p1, geo_c, geo_M, geo_mid, -1, normal.x(), normal.y(), normal.z());
				const int ell2 =
					geo::addEllipseArc(geo_mid, geo_c, geo_M, geo_p2, -1, normal.x(), normal.y(), normal.z());

				return {ell1, ell2, geo_mid};
			}
		}
	};

	const int app_cnt;
	const double anode_radius;
	const double cathode_radius;
	const double wire_radius;

	double scaled_wire_radius;
	double scaled_anode_radius;
	double cathode_mesh_size;
	double anode_mesh_size;

	std::vector<Vector3d> vertices;
	std::vector<int> geo_vertices;
	size_t vertex_cnt;
	std::vector<std::pair<int, int>> edges;
	std::vector<std::vector<int>> adj_list;

	std::vector<std::vector<Ellipse>> intersection_ellipses;
	std::vector<std::vector<VertexCurves>> vertex_curves;
	std::vector<EdgeCurves> edge_curves;
	std::vector<int> cathode_surfaces;

	bool check_cache(int ar, int cr, double quant = 100, size_t cache_size = 20) {
		std::array<int, 6> param = {app_cnt,
									static_cast<int>(std::round(anode_radius * quant)),
									static_cast<int>(std::round(cathode_radius * quant)),
									static_cast<int>(std::round(wire_radius * quant)),
									ar,
									cr};

		hash = 0;
		for (const int p : param) {
			hash ^= std::hash<int>{}(p);
		}
		hash &= 0xFFFF;
		const std::string hash_str = std::to_string(hash);

		file_name = "mesh_" + hash_str + ".msh";

		std::ifstream meta_in(std::string(PROJECT_ROOT) + "/mesh_meta_data.json");
		ordered_json meta_data = ordered_json::parse(meta_in);
		meta_in.close();

		if (meta_data.contains(hash_str)) {
			return true;
		}

		meta_data[hash_str] = json::object();
		if (meta_data.size() > cache_size) {
			auto first = meta_data.begin();
			std::string first_key = first.key();
			std::filesystem::remove(std::string(PROJECT_ROOT) + "/mesh_cache/mesh_" + hash_str + ".msh");
			meta_data.erase(first_key);
		}

		std::ofstream meta_out(std::string(PROJECT_ROOT) + "/mesh_meta_data.json");
		meta_out << meta_data.dump();
		meta_out.close();

		return false;
	}

	void get_data() {
		std::string filename = "/cathode_data/appratures_" + std::to_string(app_cnt) + ".json";

		std::ifstream file(std::string(PROJECT_ROOT) + filename);
		if (!file.is_open()) {
			std::cerr << "Error: Could not open file " << filename << std::endl;
			throw;
		}

		json json_data = json::parse(file);
		for (const auto &json_vertex : json_data["vertices"]) {
			auto v = json_vertex.get<std::vector<double>>();
			vertices.emplace_back(v[0], v[1], v[2]);
		}
		vertex_cnt = vertices.size();

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

	void find_adj_list() {
		for (size_t i = 0; i < vertex_cnt; i++) {
			adj_list.push_back(std::vector<int>());
		}

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
	}

	void find_intersection_ellipses() {
		for (size_t i = 0; i < vertex_cnt; i++) {
			const Vector3d v0 = vertices[i];
			const int geo_v0 = geo_vertices[i];
			const size_t n = adj_list[i].size();
			std::vector<Ellipse> inter_ells;
			for (size_t j = 0; j < n; j++) {
				const Vector3d v1 = vertices[adj_list[i][j]];
				const Vector3d v2 = vertices[adj_list[i][(j + 1) % n]];

				const Vector3d e1 = (v1 - v0).normalized();
				const Vector3d e2 = (v2 - v0).normalized();

				Vector3d minor = e1.cross(e2);
				if (minor.dot(v0) < 0) {
					minor *= -1;
				}
				const Vector3d major = (e1 + e2) * (scaled_wire_radius / minor.norm());
				minor.normalize();
				minor *= scaled_wire_radius;

				inter_ells.emplace_back(cathode_mesh_size, minor + v0, major + v0, v0, major.normalized(),
										minor.normalized(), std::nullopt, std::nullopt, geo_v0);
			}
			intersection_ellipses.push_back(inter_ells);
		}
	}

	void find_vertex_curves() {
		for (size_t i = 0; i < vertex_cnt; i++) {
			const Vector3d v = vertices[i];
			const int geo_v = geo_vertices[i];
			const size_t n = intersection_ellipses[i].size();
			std::vector<VertexCurves> v_curves;
			for (size_t j = 0; j < n; j++) {
				const Vector3d axis = (vertices[adj_list[i][j]] - v).normalized();
				Ellipse inter1 = intersection_ellipses[i][j];
				Ellipse inter2 = intersection_ellipses[i][(j + n - 1) % n];
				v_curves.emplace_back(cathode_mesh_size, scaled_wire_radius, axis, inter1, inter2, v, geo_v);
			}
			vertex_curves.push_back(v_curves);
		}
	}

	void find_edge_curves() {
		for (const auto &[e1, e2] : edges) {
			VertexCurves v1, v2;
			for (size_t i = 0; i < adj_list[e1].size(); i++) {
				if (adj_list[e1][i] == e2) {
					v1 = vertex_curves[e1][i];
					break;
				}
			}
			for (size_t i = 0; i < adj_list[e2].size(); i++) {
				if (adj_list[e2][i] == e1) {
					v2 = vertex_curves[e2][i];
					break;
				}
			}
			edge_curves.emplace_back(cathode_mesh_size, scaled_wire_radius, v1, v2);
		}
	}

	void make_surfaces() {
		for (const auto &edge_curve : edge_curves) {
			for (size_t i = 0; i < 10; i++) {
				cathode_surfaces.push_back(geo::addSurfaceFilling({edge_curve.loops[i]}));
			}
		}

		for (const auto &v_curves : vertex_curves) {
			const size_t n = v_curves.size();
			if (n < 5) {
				std::vector<int> loop;
				for (const auto &v_curve : v_curves) {
					loop.push_back(v_curve.curves[0]);
				}
				const int curve_loop = geo::addCurveLoop(loop);
				cathode_surfaces.push_back(geo::addSurfaceFilling({curve_loop}));
			} else {
				const int geo_vertex = v_curves[0].geo_vertex;
				const Vector3d c = v_curves[0].vertex * (1 + scaled_wire_radius);
				const int geo_c = geo::addPoint(c.x(), c.y(), c.z(), cathode_mesh_size);

				std::vector<int> cons;
				for (const auto &v_curve : v_curves) {
					cons.push_back(geo::addCircleArc(v_curve.geo_points[0], geo_vertex, geo_c));
				}

				for (size_t i = 0; i < n; i++) {
					const int loop =
						geo::addCurveLoop({v_curves[i].curves[0], cons[i], cons[(i + n - 1) % n]}, -1, true);
					cathode_surfaces.push_back(geo::addSurfaceFilling({loop}));
				}
			}

			std::vector<std::pair<int, Vector3d>> inner_points;
			for (const auto &v_curve : v_curves) {
				inner_points.emplace_back(v_curve.geo_points[2], v_curve.points[2]);
			}

			std::vector<int> lines;
			for (size_t i = 0; i < n; i++) {
				lines.push_back(geo::addLine(inner_points[i].first, inner_points[(i + 1) % n].first));
			}

			for (size_t i = 0; i < n; i++) {
				const int loop =
					geo::addCurveLoop({lines[i], v_curves[i].curves[2], v_curves[(i + 1) % n].curves[3]}, -1, true);
				cathode_surfaces.push_back(geo::addSurfaceFilling({loop}));
			}

			if (n == 3) {
				const int loop = geo::addCurveLoop(lines, -1, true);
				cathode_surfaces.push_back(geo::addPlaneSurface({loop}));
			} else if (n == 4) {
				const int diag = geo::addLine(inner_points[0].first, inner_points[2].first);
				const int loop1 = geo::addCurveLoop({lines[0], lines[1], diag}, -1, true);
				const int loop2 = geo::addCurveLoop({lines[2], lines[3], diag}, -1, true);
				cathode_surfaces.push_back(geo::addPlaneSurface({loop1}));
				cathode_surfaces.push_back(geo::addPlaneSurface({loop2}));
			} else {
				Vector3d center = Vector3d::Zero();
				for (const auto &[_, p] : inner_points) {
					center += p;
				}
				center /= n;
				const int geo_c = geo::addPoint(center.x(), center.y(), center.z());

				std::vector<int> more_lines;
				for (const auto &[geo_p, _] : inner_points) {
					more_lines.push_back(geo::addLine(geo_p, geo_c));
				}

				for (size_t i = 0; i < n; i++) {
					int loop = geo::addCurveLoop({lines[i], more_lines[i], more_lines[(i + 1) % n]}, -1, true);
					cathode_surfaces.push_back(geo::addPlaneSurface({loop}));
				}
			}
		}
	}

	std::pair<int, std::vector<int>> make_anode() const {
		const int o = geo::addPoint(0.0, 0.0, 0.0, anode_mesh_size);

		const int zp = geo::addPoint(0.0, 0.0, scaled_anode_radius, anode_mesh_size);
		const int zm = geo::addPoint(0.0, 0.0, -scaled_anode_radius, anode_mesh_size);
		const int yp = geo::addPoint(0.0, scaled_anode_radius, 0.0, anode_mesh_size);
		const int ym = geo::addPoint(0.0, -scaled_anode_radius, 0.0, anode_mesh_size);
		const int xp = geo::addPoint(scaled_anode_radius, 0.0, 0.0, anode_mesh_size);
		const int xm = geo::addPoint(-scaled_anode_radius, 0.0, 0.0, anode_mesh_size);

		const int zpyp = geo::addCircleArc(zp, o, yp);
		const int zpym = geo::addCircleArc(zp, o, ym);
		const int zpxp = geo::addCircleArc(zp, o, xp);
		const int zpxm = geo::addCircleArc(zp, o, xm);

		const int zmyp = geo::addCircleArc(zm, o, yp);
		const int zmym = geo::addCircleArc(zm, o, ym);
		const int zmxp = geo::addCircleArc(zm, o, xp);
		const int zmxm = geo::addCircleArc(zm, o, xm);

		const int ypxp = geo::addCircleArc(yp, o, xp);
		const int ypxm = geo::addCircleArc(yp, o, xm);
		const int ymxp = geo::addCircleArc(ym, o, xp);
		const int ymxm = geo::addCircleArc(ym, o, xm);

		std::array<int, 8> l;
		l[0] = geo::addCurveLoop({zpyp, ypxp, zpxp}, -1, true);
		l[1] = geo::addCurveLoop({zpyp, ypxm, zpxm}, -1, true);
		l[2] = geo::addCurveLoop({zpym, ymxp, zpxp}, -1, true);
		l[3] = geo::addCurveLoop({zpym, ymxm, zpxm}, -1, true);
		l[4] = geo::addCurveLoop({zmyp, ypxp, zmxp}, -1, true);
		l[5] = geo::addCurveLoop({zmyp, ypxm, zmxm}, -1, true);
		l[6] = geo::addCurveLoop({zmym, ymxp, zmxp}, -1, true);
		l[7] = geo::addCurveLoop({zmym, ymxm, zmxm}, -1, true);

		std::vector<int> s(8);
		for (size_t i = 0; i < 8; i++) {
			s[i] = geo::addSurfaceFilling({l[i]});
		}

		return {geo::addSurfaceLoop(s), s};
	}

	void apply_mesh_settings(const std::vector<int> &cathode_surfaces, int app_cnt, double wire_radius,
							 double fine_size, double coarse_size, double scale = 0.75, double dist = 20.0) {
		int dist_field = gmsh::model::mesh::field::add("Distance");

		// Convert surface tags to doubles
		std::vector<double> surface_tags;
		for (int tag : cathode_surfaces) {
			surface_tags.push_back(static_cast<double>(tag));
		}
		gmsh::model::mesh::field::setNumbers(dist_field, "FacesList", surface_tags);

		int thresh_field = gmsh::model::mesh::field::add("Threshold");
		gmsh::model::mesh::field::setNumber(thresh_field, "IField", static_cast<double>(dist_field));
		gmsh::model::mesh::field::setNumber(thresh_field, "LcMin", fine_size);
		gmsh::model::mesh::field::setNumber(thresh_field, "LcMax", coarse_size);
		gmsh::model::mesh::field::setNumber(thresh_field, "DistMin", 0.0);
		const double adj_dist = std::max(1.0, (dist / std::sqrt(static_cast<double>(app_cnt))));
		gmsh::model::mesh::field::setNumber(thresh_field, "DistMax", wire_radius * adj_dist);
		gmsh::model::mesh::field::setNumber(thresh_field, "Sigmoid", scale);

		gmsh::model::mesh::field::setAsBackgroundMesh(thresh_field);

		gmsh::option::setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0);
		gmsh::option::setNumber("Mesh.CharacteristicLengthFromPoints", 0);
		gmsh::option::setNumber("Mesh.CharacteristicLengthFromCurvature", 0);
	}

	void print_mesh_statistics() const {
		// Count all nodes in the mesh
		std::vector<std::size_t> nodeTags;
		std::vector<double> coord;
		std::vector<double> parametricCoord;
		gmsh::model::mesh::getNodes(nodeTags, coord, parametricCoord, -1, -1, true, false);
		const size_t numNodes = nodeTags.size();

		// Get tetrahedra (3D elements)
		std::vector<int> elementTypes;
		std::vector<std::vector<std::size_t>> elementTags;
		std::vector<std::vector<std::size_t>> elementNodeTags;
		gmsh::model::mesh::getElements(elementTypes, elementTags, elementNodeTags, 3);

		size_t numTets = 0;
		std::set<std::pair<std::size_t, std::size_t>> allEdges;

		// Process tetrahedra: count and extract edges
		for (size_t i = 0; i < elementTypes.size(); i++) {
			if (elementTypes[i] == 4) { // Tetrahedra
				numTets = elementTags[i].size();

				// Extract edges from tetrahedra
				for (size_t j = 0; j < numTets; j++) {
					size_t baseIdx = j * 4;

					// 6 edges per tet: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
					for (int e = 0; e < 6; e++) {
						size_t n1, n2;
						switch (e) {
						case 0:
							n1 = elementNodeTags[i][baseIdx];
							n2 = elementNodeTags[i][baseIdx + 1];
							break;
						case 1:
							n1 = elementNodeTags[i][baseIdx];
							n2 = elementNodeTags[i][baseIdx + 2];
							break;
						case 2:
							n1 = elementNodeTags[i][baseIdx];
							n2 = elementNodeTags[i][baseIdx + 3];
							break;
						case 3:
							n1 = elementNodeTags[i][baseIdx + 1];
							n2 = elementNodeTags[i][baseIdx + 2];
							break;
						case 4:
							n1 = elementNodeTags[i][baseIdx + 1];
							n2 = elementNodeTags[i][baseIdx + 3];
							break;
						case 5:
							n1 = elementNodeTags[i][baseIdx + 2];
							n2 = elementNodeTags[i][baseIdx + 3];
							break;
						}
						allEdges.insert({std::min(n1, n2), std::max(n1, n2)});
					}
				}
			}
		}

		// Get triangles (2D surface elements)
		elementTypes.clear();
		elementTags.clear();
		elementNodeTags.clear();
		gmsh::model::mesh::getElements(elementTypes, elementTags, elementNodeTags, 2);

		size_t numTriangles = 0;
		std::set<std::pair<std::size_t, std::size_t>> surfaceEdges;
		std::set<std::size_t> surfaceNodes;

		// Process triangles: count and extract surface edges and nodes
		for (size_t i = 0; i < elementTypes.size(); i++) {
			if (elementTypes[i] == 2) { // Triangles
				numTriangles += elementTags[i].size();

				// Extract edges and nodes from triangles
				for (size_t j = 0; j < elementTags[i].size(); j++) {
					size_t baseIdx = j * 3;

					// Collect nodes from this triangle
					for (int k = 0; k < 3; k++) {
						surfaceNodes.insert(elementNodeTags[i][baseIdx + k]);
					}

					// 3 edges per triangle: (0,1), (1,2), (2,0)
					for (int e = 0; e < 3; e++) {
						size_t n1 = elementNodeTags[i][baseIdx + e];
						size_t n2 = elementNodeTags[i][baseIdx + ((e + 1) % 3)];
						surfaceEdges.insert({std::min(n1, n2), std::max(n1, n2)});
						allEdges.insert({std::min(n1, n2), std::max(n1, n2)});
					}
				}
			}
		}

		// Calculate proportions
		double nodeSurfaceProportion = (numNodes > 0) ? (100.0 * surfaceNodes.size() / numNodes) : 0.0;
		double edgeSurfaceProportion = (allEdges.size() > 0) ? (100.0 * surfaceEdges.size() / allEdges.size()) : 0.0;
		double cellSurfaceProportion = (numTriangles > 0) ? (100.0 * numTriangles / numTets) : 0.0;

		// Print results
		std::cout << "\n=== Mesh Statistics ===" << std::endl;
		std::cout << "\n--- Nodes ---" << std::endl;
		std::cout << "Total nodes: " << numNodes << std::endl;
		std::cout << "Surface nodes: " << surfaceNodes.size() << " (" << std::fixed << std::setprecision(1)
				  << nodeSurfaceProportion << "%)" << std::endl;

		std::cout << "\n--- Edges ---" << std::endl;
		std::cout << "Total edges: " << allEdges.size() << std::endl;
		std::cout << "Surface edges: " << surfaceEdges.size() << " (" << std::fixed << std::setprecision(1)
				  << edgeSurfaceProportion << "%)" << std::endl;

		std::cout << "\n--- Elements ---" << std::endl;
		std::cout << "Tetrahedra: " << numTets << std::endl;
		std::cout << "Triangles: " << numTriangles << " (" << std::fixed << std::setprecision(1)
				  << cellSurfaceProportion << "%)" << std::endl;

		std::cout << std::resetiosflags(std::ios::fixed) << std::setprecision(16);

		std::cout << "\n--- Physical Groups ---" << std::endl;
		std::vector<std::pair<int, int>> physicalGroups;
		gmsh::model::getPhysicalGroups(physicalGroups);
		for (auto [dim, tag] : physicalGroups) {
			std::string name;
			gmsh::model::getPhysicalName(dim, tag, name);
			std::vector<int> entities;
			gmsh::model::getEntitiesForPhysicalGroup(dim, tag, entities);
			std::cout << "Physical group '" << name << "' (dim=" << dim << ", tag=" << tag << "): " << entities.size()
					  << " entities" << std::endl;
		}
		std::cout << "======================\n" << std::endl;
	}

	void load_cathode_data() {
		std::vector<size_t> nodes;
		std::vector<double> cords;
		gmsh::model::mesh::getNodesForPhysicalGroup(2, 1, nodes, cords);

		std::unordered_map<size_t, size_t> node_index;
		std::vector<double> X(nodes.size()), Y(nodes.size()), Z(nodes.size());
		std::vector<Vector3d> mesh_nodes;

		for (size_t i = 0; i < nodes.size(); i++) {
			const Vector3d c(cords[3 * i] * cathode_radius, cords[3 * i + 1] * cathode_radius,
							 cords[3 * i + 2] * cathode_radius);

			mesh_nodes.push_back(c);
			X[i] = c.x();
			Y[i] = c.y();
			Z[i] = c.z();
			node_index[nodes[i]] = i;
		}

		std::vector<int> cathode_entitites;
		gmsh::model::getEntitiesForPhysicalGroup(2, 1, cathode_entitites);
		std::vector<size_t> I, J, K;
		for (const auto &entity : cathode_entitites) {
			std::vector<size_t> elements, e_nodes;
			gmsh::model::mesh::getElementsByType(2, elements, e_nodes, entity);
			for (size_t i = 0; i < elements.size(); i++) {
				I.push_back(node_index[e_nodes[3 * i]]);
				J.push_back(node_index[e_nodes[3 * i + 1]]);
				K.push_back(node_index[e_nodes[3 * i + 2]]);
			}
		}

		std::ifstream meta_in(std::string(PROJECT_ROOT) + "/mesh_meta_data.json");
		ordered_json meta_data = ordered_json::parse(meta_in);
		meta_in.close();

		if (!meta_data[std::to_string(hash)].contains("app_cnt")) {
			meta_data[std::to_string(hash)] = {{"app_cnt", app_cnt},
											   {"anode_radius", anode_radius},
											   {"cathode_radius", cathode_radius},
											   {"wire_radius", wire_radius},
											   {"X", X},
											   {"Y", Y},
											   {"Z", Z},
											   {"I", I},
											   {"J", J},
											   {"K", K}};
			std::ofstream meta_out(std::string(PROJECT_ROOT) + "/mesh_meta_data.json");
			meta_out << meta_data.dump();
			meta_out.close();
		}
	}
};

#endif
