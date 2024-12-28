use nalgebra::{Matrix3, Vector3};
use stl::{self, Triangle};

const A_R: f64 = 0.05;
const C_R: f64 = 0.5;
const V: f64 = 1e5;

fn main() {
    let mut file = std::fs::File::open("../anode_meshes/appratures_6.stl").unwrap();
    let stl_mesh = stl::read_stl(&mut file).unwrap();
    let mesh = my_mesh(&stl_mesh.triangles, 0.05);
    let (c, a) = ca(&mesh);
    let (hb, lb, _) = bounds(&mesh);
    let e = |p: &Vector3<f64>| -> Vector3<f64> { e_feild(p, &c, &a) };
}

fn av3<T: Into<f64> + Copy>(v: &[T; 3]) -> Vector3<f64> {
    Vector3::new(v[0].into(), v[1].into(), v[2].into())
}

fn v3a<T: Into<f64> + Copy>(v: &Vector3<T>) -> [f64; 3] {
    [v.x as f64, v.y as f64, v.z as f64]
}

fn my_mesh(tris: &Vec<Triangle>, scale: f64) -> Vec<[Vector3<f64>; 3]> {
    let mut mesh: Vec<[Vector3<f64>; 3]> = Vec::new();
    for tri in tris {
        let v1: Vector3<f64> = av3(&tri.v1);
        let v2: Vector3<f64> = av3(&tri.v2);
        let v3: Vector3<f64> = av3(&tri.v3);

        mesh.push([v1 * scale, v2 * scale, v3 * scale])
    }

    return mesh;
}

fn bounds(mesh: &Vec<[Vector3<f64>; 3]>) -> (f64, f64, Vector3<f64>) {
    let mut h = 0.0;
    let mut l = f64::MAX;
    let mut sum: Vector3<f64> = Vector3::new(0.0, 0.0, 0.0);
    let mut v_cnt = 0.0;
    for tri in mesh {
        for v in tri {
            sum += v;
            v_cnt += 1.0;
            let n = v.norm();
            if n < l {
                l = n
            }
            if n > h {
                h = n
            }
        }
    }

    return (l, h, sum / v_cnt);
}

fn ca(mesh: &Vec<[Vector3<f64>; 3]>) -> (Vec<Vector3<f64>>, Vec<f64>) {
    let mut centroids: Vec<Vector3<f64>> = Vec::new();
    let mut areas: Vec<f64> = Vec::new();
    let mut area = 0.0;
    for tri in mesh {
        let c: Vector3<f64> = (tri[0] + tri[1] + tri[2]) / 3.0;
        let a = (tri[1] - tri[0]).cross(&(tri[2] - tri[0])).norm() / 2.0;
        centroids.push(c);
        areas.push(a);
        area += a;
    }

    let e_c = 47917944.84 * V / (1.0 / A_R - 1.0 / C_R);
    for a in areas.iter_mut() {
        *a *= e_c / area;
    }

    return (centroids, areas);
}

fn e_feild(r: &Vector3<f64>, c: &Vec<Vector3<f64>>, a: &Vec<f64>) -> Vector3<f64> {
    let mut e_tot: Vector3<f64> = Vector3::new(0.0, 0.0, 0.0);
    let mut d: Vector3<f64>;
    for i in 0..c.len() {
        d = r - c[i];
        e_tot -= a[i] * d / d.norm().powi(3)
    }
    return e_tot;
}

const A_43: [f64; 6] = [
    1250f64 / 9801f64,
    -5935429f64 / 44800000f64,
    10989429f64 / 44800000f64,
    949f64 / 5700f64,
    3267f64 / 11900f64,
    400f64 / 6783f64,
];

const B1_43: [f64; 3] = [949f64 / 5700f64, 3267f64 / 11900f64, 400f64 / 6783f64];

const B2_43: [f64; 4] = [
    949f64 / 5700f64,
    323433f64 / 583100f64,
    16000f64 / 142443f64,
    514f64 / 3087f64,
];

const B3_43: [f64; 4] = [
    4483f64 / 40000f64,
    11517f64 / 40000f64,
    3f64 / 20f64,
    -1f64 / 20f64,
];

const B4_43: [f64; 4] = [
    13399f64 / 95000f64,
    -153549f64 / 595000f64,
    2096f64 / 2261f64,
    19f64 / 100f64,
];

fn rkn43<F>(f: &F, u: &mut Vector3<f64>, v: &mut Vector3<f64>, tol: f64, mut h: f64)
where
    F: Fn(&Vector3<f64>) -> Vector3<f64>,
{
    let f1: Vector3<f64> = f(u);
    let f2: Vector3<f64> = f(&(*u + h * A_43[0] * f1));
    let f3: Vector3<f64> = f(&(*u + h * (A_43[1] * f1 + A_43[2] * f2)));
    let f4: Vector3<f64> = f(&(*u + h * (A_43[3] * f1 + A_43[4] * f2 + A_43[5] * f3)));

    let u_hat: Vector3<f64> =
        *u + h * (*v + h * (B3_43[0] * f1 + B3_43[1] * f2 + B3_43[2] * f3 + B3_43[3] * f4));
    *u += h * (*v + h * (B1_43[0] * f1 + B1_43[1] * f2 + B1_43[2] * f3));

    let v_hat: Vector3<f64> =
        *v + h * (B4_43[0] * f1 + B4_43[1] * f2 + B4_43[2] * f3 + B4_43[3] * f4);
    *v += h * (B2_43[0] * f1 + B2_43[1] * f2 + B2_43[2] * f3 + B2_43[3] * f4);

    let vd: Vector3<f64> = *v - v_hat;
    let ud: Vector3<f64> = *u - u_hat;
    let n = vd.amax().max(ud.amax());
    h *= 0.9 * (tol / n).powf(0.25);
}

fn intersect(p1: &Vector3<f64>, p2: &Vector3<f64>, v: &[Vector3<f64>; 3]) -> bool {
    let d: Vector3<f64> = p1 - p2;
    let e1: Vector3<f64> = v[1] - v[0];
    let e2: Vector3<f64> = v[2] - v[1];

    let mut m: Matrix3<f64> = Matrix3::from_columns(&[-d, e1, e2]);
    if m.try_inverse_mut() {
        let s: Vector3<f64> = m * (p1 - v[0]);
        if (0.0 <= s.x) && (s.x <= 1.0) && (0.0 <= s.y) && (0.0 <= s.z) && ((s.y + s.x) <= 1.0) {
            return true;
        }
    }

    return false;
}

fn find_path<Solver, F>(
    mesh: &Vec<[Vector3<f64>; 3]>,
    e: F,
    b: (f64, f64),
    max_orbit: u8,
    initial: (Vector3<f64>, Vector3<f64>),
    solver: Solver,
    tol: f64,
    mut h: f64,
) -> (u8, Vector3<f64>)
where
    Solver: Fn(&F, &mut Vector3<f64>, &mut Vector3<f64>, f64, f64),
    F: Fn(&Vector3<f64>) -> Vector3<f64>,
{
    let mut u1: Vector3<f64> = initial.0;
    let mut u: Vector3<f64> = initial.0;
    let mut v: Vector3<f64> = initial.1;

    let mut n2 = u.norm();
    let mut n1 = u.norm();

    let mut n: f64;
    let mut orbit_cnt: u8 = 0;
    loop {
        solver(&e, &mut u, &mut v, tol, h);
        n = u.norm();

        if (n2 < n1) && (n < n1) && (b.1 < n1) {
            orbit_cnt += 1;
            if orbit_cnt >= max_orbit {
                break;
            }
        }

        if (n.min(n1) <= b.1) || (n.max(n1) >= b.0) {
            for tri in mesh {
                if intersect(&u1, &u, tri) {
                    break;
                }
            }
        }

        n2 = n1;
        n1 = n;
        u1 = u;
    }

    return (orbit_cnt, u);
}
