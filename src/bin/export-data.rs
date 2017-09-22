extern crate docopt;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate bincode;
extern crate petgraph;
extern crate rand;
extern crate statrs;
extern crate capngraph;

use docopt::Docopt;
use std::fs::File;
use petgraph::prelude::*;
use std::io::{BufWriter, Write};

#[cfg_attr(rustfmt, rustfmt_skip)]
const USAGE: &str = "
Export costs/benefits and edge weights to a text format for use with SSA / BCT. IMM requires an
additional conversion step beyond this.  

Usage:
    export-data unweighted <graph> <out>
    export-data cost-aware <graph> <costs> <out>
    export-data ctvm <graph> <costs> <benefits> <out>
    export-data (-h | --help)

Options:
    -h --help           Show this screen.
";

#[derive(Serialize, Deserialize, Debug)]
struct Args {
    arg_graph: String,
    arg_costs: Option<String>,
    arg_benefits: Option<String>,
    arg_out: String,
}

fn main() {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    let g = capngraph::load_graph(&args.arg_graph).unwrap();
    let costs: Vec<f64> = args.arg_costs
        .as_ref()
        .map(|costs| {
            bincode::deserialize_from(&mut File::open(costs).unwrap(), bincode::Infinite).unwrap()
        })
        .unwrap_or_else(|| vec![1f64; g.node_count()]);
    assert_eq!(costs.len(), g.node_count());

    let benefits: Vec<f64> = args.arg_benefits
        .as_ref()
        .map(|benefits| {
            bincode::deserialize_from(&mut File::open(benefits).unwrap(), bincode::Infinite)
                .unwrap()
        })
        .unwrap_or_else(|| vec![1f64; g.node_count()]);
    assert_eq!(benefits.len(), g.node_count());

    let out = File::create(&args.arg_out).unwrap();
    let mut writer = BufWriter::new(out);

    writeln!(writer, "{} {}", g.node_count(), g.edge_count()).unwrap();

    for i in 0..g.node_count() {
        writeln!(writer, "{} {} {}", i, costs[i], benefits[i]).unwrap();
    }

    for edge in g.edge_references() {
        writeln!(writer,
                 "{} {} {}",
                 edge.source().index(),
                 edge.target().index(),
                 f64::from(*edge.weight()))
            .unwrap();
    }
}
