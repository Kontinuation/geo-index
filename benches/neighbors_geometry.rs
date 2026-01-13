//! Benchmarks for the `neighbors_geometry` method.
//!
//! This benchmark suite measures the throughput of geometry-based nearest neighbor searches
//! across different combinations of indexed and query geometry types.

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
    Throughput,
};
use geo_0_31::algorithm::BoundingRect;
use geo_0_31::{coord, Geometry, LineString, Point, Polygon};
use geo_index::rtree::distance::{EuclideanDistance, SliceGeometryAccessor};
use geo_index::rtree::sort::HilbertSort;
use geo_index::rtree::{RTreeBuilder, RTreeIndex};
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

/// Configuration for benchmark runs
struct BenchConfig {
    /// Number of indexed geometries
    num_geometries: usize,
    /// Number of vertices in polygons
    num_vertices: usize,
    /// Number of neighbors to find
    k: usize,
    /// Seed for reproducible random generation
    seed: u64,
}

impl BenchConfig {
    fn new(num_geometries: usize, num_vertices: usize) -> Self {
        Self {
            num_geometries,
            num_vertices,
            k: 10,
            seed: 42,
        }
    }
}

/// Generate a random point within bounds [0, 1000]
fn generate_random_point<R: Rng>(rng: &mut R) -> Point<f64> {
    let x_dist = Uniform::new(0.0, 1000.0);
    let y_dist = Uniform::new(0.0, 1000.0);
    Point::new(rng.sample(x_dist), rng.sample(y_dist))
}

/// Generate a random polygon with the specified number of vertices
fn generate_random_polygon<R: Rng>(rng: &mut R, num_vertices: usize) -> Polygon<f64> {
    let size_dist = Uniform::new(5.0, 20.0);
    let half_size = rng.sample(size_dist) / 2.0;

    // Center position ensuring polygon fits in bounds
    let center_x_dist = Uniform::new(half_size + 10.0, 1000.0 - half_size - 10.0);
    let center_y_dist = Uniform::new(half_size + 10.0, 1000.0 - half_size - 10.0);
    let center_x = rng.sample(center_x_dist);
    let center_y = rng.sample(center_y_dist);

    let num_vertices = num_vertices.max(3);
    let mut coords = Vec::with_capacity(num_vertices + 1);

    let start_angle_dist = Uniform::new(0.0, 2.0 * PI);
    let mut angle: f64 = rng.sample(start_angle_dist);
    let dangle = 2.0 * PI / num_vertices as f64;

    // Add some randomness to radii for more realistic shapes
    let radius_variation = Uniform::new(0.7, 1.3);

    for _ in 0..num_vertices {
        let r = half_size * rng.sample(radius_variation);
        coords.push(coord! {
            x: angle.cos() * r + center_x,
            y: angle.sin() * r + center_y,
        });
        angle += dangle;
    }
    // Close the ring
    coords.push(coords[0]);

    Polygon::new(LineString::from(coords), vec![])
}

/// Generate random points as geometries
fn generate_points(seed: u64, count: usize) -> Vec<Geometry<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| Geometry::Point(generate_random_point(&mut rng)))
        .collect()
}

/// Generate random polygons as geometries
fn generate_polygons(seed: u64, count: usize, num_vertices: usize) -> Vec<Geometry<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| Geometry::Polygon(generate_random_polygon(&mut rng, num_vertices)))
        .collect()
}

/// Build an RTree from geometries
fn build_rtree(geometries: &[Geometry<f64>]) -> geo_index::rtree::RTree<f64> {
    let mut builder = RTreeBuilder::new(geometries.len() as u32);
    for geom in geometries {
        if let Some(rect) = geom.bounding_rect() {
            builder.add(rect.min().x, rect.min().y, rect.max().x, rect.max().y);
        } else {
            builder.add(0.0, 0.0, 0.0, 0.0);
        }
    }
    builder.finish::<HilbertSort>()
}

/// Benchmark a specific geometry type combination
fn bench_geometry_combination(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    config: &BenchConfig,
    indexed_geometries: &[Geometry<f64>],
    query_geometries: &[Geometry<f64>],
) {
    let tree = build_rtree(indexed_geometries);
    let metric = EuclideanDistance;
    let accessor = SliceGeometryAccessor::new(indexed_geometries);

    group.throughput(Throughput::Elements(query_geometries.len() as u64));

    group.bench_with_input(
        BenchmarkId::new(name, config.num_geometries),
        &config.num_geometries,
        |b, _| {
            b.iter(|| {
                for query_geom in query_geometries {
                    let _ = tree.neighbors_geometry(
                        query_geom,
                        Some(config.k),
                        None,
                        &metric,
                        &accessor,
                    );
                }
            })
        },
    );
}

/// Benchmark point index with point query
fn benchmark_point_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors_geometry/point_index_point_query");

    let geometry_counts = vec![100, 1000, 10000];
    let num_queries = 100;

    for num_geometries in geometry_counts {
        let config = BenchConfig::new(num_geometries, 0); // vertices not used for points

        let indexed_geometries = generate_points(config.seed, config.num_geometries);
        let query_geometries = generate_points(config.seed + 1000, num_queries);

        bench_geometry_combination(
            &mut group,
            "n",
            &config,
            &indexed_geometries,
            &query_geometries,
        );
    }

    group.finish();
}

/// Benchmark point index with polygon query
fn benchmark_point_polygon(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors_geometry/point_index_polygon_query");

    let geometry_counts = vec![100, 1000, 10000];
    let vertex_counts = vec![4, 8, 16, 32];
    let num_queries = 100;

    for num_geometries in &geometry_counts {
        for num_vertices in &vertex_counts {
            let config = BenchConfig::new(*num_geometries, *num_vertices);

            let indexed_geometries = generate_points(config.seed, config.num_geometries);
            let query_geometries =
                generate_polygons(config.seed + 1000, num_queries, config.num_vertices);

            let name = format!("n={}/v={}", num_geometries, num_vertices);
            let tree = build_rtree(&indexed_geometries);
            let metric = EuclideanDistance;
            let accessor = SliceGeometryAccessor::new(&indexed_geometries);

            group.throughput(Throughput::Elements(num_queries as u64));
            group.bench_with_input(BenchmarkId::from_parameter(&name), &name, |b, _| {
                b.iter(|| {
                    for query_geom in &query_geometries {
                        let _ = tree.neighbors_geometry(
                            query_geom,
                            Some(config.k),
                            None,
                            &metric,
                            &accessor,
                        );
                    }
                })
            });
        }
    }

    group.finish();
}

/// Benchmark polygon index with point query
fn benchmark_polygon_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors_geometry/polygon_index_point_query");

    let geometry_counts = vec![100, 1000, 10000];
    let vertex_counts = vec![4, 8, 16, 32];
    let num_queries = 100;

    for num_geometries in &geometry_counts {
        for num_vertices in &vertex_counts {
            let config = BenchConfig::new(*num_geometries, *num_vertices);

            let indexed_geometries =
                generate_polygons(config.seed, config.num_geometries, config.num_vertices);
            let query_geometries = generate_points(config.seed + 1000, num_queries);

            let name = format!("n={}/v={}", num_geometries, num_vertices);
            let tree = build_rtree(&indexed_geometries);
            let metric = EuclideanDistance;
            let accessor = SliceGeometryAccessor::new(&indexed_geometries);

            group.throughput(Throughput::Elements(num_queries as u64));
            group.bench_with_input(BenchmarkId::from_parameter(&name), &name, |b, _| {
                b.iter(|| {
                    for query_geom in &query_geometries {
                        let _ = tree.neighbors_geometry(
                            query_geom,
                            Some(config.k),
                            None,
                            &metric,
                            &accessor,
                        );
                    }
                })
            });
        }
    }

    group.finish();
}

/// Benchmark polygon index with polygon query
fn benchmark_polygon_polygon(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors_geometry/polygon_index_polygon_query");

    let geometry_counts = vec![100, 1000, 10000];
    let vertex_counts = vec![4, 8, 16, 32];
    let num_queries = 100;

    for num_geometries in &geometry_counts {
        for num_vertices in &vertex_counts {
            let config = BenchConfig::new(*num_geometries, *num_vertices);

            let indexed_geometries =
                generate_polygons(config.seed, config.num_geometries, config.num_vertices);
            let query_geometries =
                generate_polygons(config.seed + 1000, num_queries, config.num_vertices);

            let name = format!("n={}/v={}", num_geometries, num_vertices);
            let tree = build_rtree(&indexed_geometries);
            let metric = EuclideanDistance;
            let accessor = SliceGeometryAccessor::new(&indexed_geometries);

            group.throughput(Throughput::Elements(num_queries as u64));
            group.bench_with_input(BenchmarkId::from_parameter(&name), &name, |b, _| {
                b.iter(|| {
                    for query_geom in &query_geometries {
                        let _ = tree.neighbors_geometry(
                            query_geom,
                            Some(config.k),
                            None,
                            &metric,
                            &accessor,
                        );
                    }
                })
            });
        }
    }

    group.finish();
}

/// Benchmark scaling with number of neighbors (k)
fn benchmark_k_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors_geometry/k_scaling");

    let k_values = vec![1, 5, 10, 25, 50, 100];
    let num_geometries = 10000;
    let num_vertices = 8;
    let num_queries = 100;
    let seed = 42u64;

    let indexed_geometries = generate_polygons(seed, num_geometries, num_vertices);
    let query_geometries = generate_polygons(seed + 1000, num_queries, num_vertices);
    let tree = build_rtree(&indexed_geometries);
    let metric = EuclideanDistance;
    let accessor = SliceGeometryAccessor::new(&indexed_geometries);

    for k in k_values {
        group.throughput(Throughput::Elements(num_queries as u64));
        group.bench_with_input(BenchmarkId::new("k", k), &k, |b, &k| {
            b.iter(|| {
                for query_geom in &query_geometries {
                    let _ = tree.neighbors_geometry(query_geom, Some(k), None, &metric, &accessor);
                }
            })
        });
    }

    group.finish();
}

/// Benchmark comparison: point-based neighbors vs geometry-based neighbors_geometry
fn benchmark_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors_geometry/vs_point_neighbors");

    let num_geometries = 10000;
    let num_queries = 100;
    let k = 10;
    let seed = 42u64;

    // Generate point geometries for fair comparison
    let indexed_geometries = generate_points(seed, num_geometries);
    let query_points: Vec<Point<f64>> = {
        let mut rng = StdRng::seed_from_u64(seed + 1000);
        (0..num_queries)
            .map(|_| generate_random_point(&mut rng))
            .collect()
    };
    let query_geometries: Vec<Geometry<f64>> =
        query_points.iter().map(|p| Geometry::Point(*p)).collect();

    let tree = build_rtree(&indexed_geometries);
    let metric = EuclideanDistance;
    let accessor = SliceGeometryAccessor::new(&indexed_geometries);

    group.throughput(Throughput::Elements(num_queries as u64));

    // Benchmark point-based neighbors
    group.bench_function("neighbors", |b| {
        b.iter(|| {
            for point in &query_points {
                let _ = tree.neighbors(point.x(), point.y(), Some(k), None);
            }
        })
    });

    // Benchmark neighbors_with_distance (uses distance metric)
    group.bench_function("neighbors_with_distance", |b| {
        b.iter(|| {
            for point in &query_points {
                let _ = tree.neighbors_with_distance(point.x(), point.y(), Some(k), None, &metric);
            }
        })
    });

    // Benchmark geometry-based neighbors_geometry (with point geometries)
    group.bench_function("neighbors_geometry_point", |b| {
        b.iter(|| {
            for query_geom in &query_geometries {
                let _ = tree.neighbors_geometry(query_geom, Some(k), None, &metric, &accessor);
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_point_point,
    benchmark_point_polygon,
    benchmark_polygon_point,
    benchmark_polygon_polygon,
    benchmark_k_scaling,
    benchmark_comparison,
);
criterion_main!(benches);
