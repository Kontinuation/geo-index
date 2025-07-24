use bytemuck::cast_slice;
use clap::{Parser, ValueEnum};
use geo_index::rtree::sort::{HilbertSort, STRSort};
use geo_index::rtree::{RTree, RTreeBuilder, RTreeIndex};
use geo_index::IndexableNum;
use geo_types::Polygon;
use std::fs::read;
use std::fs::File;
use std::io::{self, Write};
use wkt::ToWkt;

#[derive(Parser, Debug)]
#[command(author, version, about = "Tool for dumping bounding boxes from a spatial index.")]
struct Args {
    /// Tree type: hilbert or str
    #[arg(short = 't', long, value_enum)]
    tree_type: TreeType,

    /// Path to input .raw file
    input: String,

    /// Layer to dump. the base layer is 0, the next layer is 1, etc.
    #[arg(short = 'l', long, default_value_t = 0)]
    layer: usize,

    /// Output file (WKT), if not present, print to stdout
    #[arg(short = 'o', long)]
    output: Option<String>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum TreeType {
    Hilbert,
    Str,
}

/// Load bounding box data from a raw file
fn load_dataset(file_path: &str) -> Vec<f64> {
    let buf = read(file_path).unwrap_or_else(|_| panic!("Failed to read {}", file_path));
    cast_slice(&buf).to_vec()
}

fn box_to_wkt(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> String {
    let poly = Polygon::new(
        vec![
            (min_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y),
            (min_x, min_y),
        ]
        .into(),
        vec![],
    );
    poly.to_wkt().to_string()
}

/// Construct an RTree with Hilbert sorting
fn construct_rtree_hilbert<N: IndexableNum>(boxes_buf: &[N]) -> RTree<N> {
    let mut builder = RTreeBuilder::new((boxes_buf.len() / 4) as _);
    for box_ in boxes_buf.chunks(4) {
        let min_x = box_[0];
        let min_y = box_[1];
        let max_x = box_[2];
        let max_y = box_[3];
        builder.add(min_x, min_y, max_x, max_y);
    }
    builder.finish::<HilbertSort>()
}

/// Construct an RTree with STR sorting
fn construct_rtree_str<N: IndexableNum>(boxes_buf: &[N]) -> RTree<N> {
    let mut builder = RTreeBuilder::new((boxes_buf.len() / 4) as _);
    for box_ in boxes_buf.chunks(4) {
        let min_x = box_[0];
        let min_y = box_[1];
        let max_x = box_[2];
        let max_y = box_[3];
        builder.add(min_x, min_y, max_x, max_y);
    }
    builder.finish::<STRSort>()
}

fn main() {
    let args = Args::parse();
    let mut writer: Box<dyn Write> = match &args.output {
        Some(path) => Box::new(File::create(path).expect("Failed to create output file")),
        None => Box::new(io::stdout()),
    };

    let data = load_dataset(&args.input);
    let index = match args.tree_type {
        TreeType::Hilbert => construct_rtree_hilbert(&data),
        TreeType::Str => construct_rtree_str(&data),
    };

    println!("index: {:?}", index.metadata());
    writeln!(writer, "bbox").unwrap();
    for box_ in index.boxes_at_level(args.layer).unwrap().chunks(4) {
        writeln!(
            writer,
            "\"{}\"",
            box_to_wkt(box_[0], box_[1], box_[2], box_[3])
        )
        .unwrap();
    }
}
