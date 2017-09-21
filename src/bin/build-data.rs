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

#[cfg_attr(rustfmt, rustfmt_skip)]
const USAGE: &str =
    "
Construct cost/benefit data for the TipTop implementation.

The `constant` command should in general be avoided, but is provided because it can simplify
experiment pipelines.

The `uniform` and `out-degree` correspond to those settings from the experimental section of the
paper.

The `binary-uniform` setting corresponds to the full CTVM setting, where <on-fraction> sets the
fraction of nodes targeted and <min>, <max> sets the distribution of benefits for users that are
active.

Usage:
    build-data constant <graph> <out>
    build-data uniform <graph> <out> <min> <max>
    build-data out-degree <graph> <out> (--linear | --log)
    build-data binary-uniform <graph> <out> <on-fraction> <min> <max>
    build-data (-h | --help)

Options:
    -h --help           Show this screen.
    --linear            Use linear scaling.
    --log               Use log (i.e. ln()) scaling.
";

#[derive(Serialize, Deserialize, Debug)]
struct Args {
    arg_graph: String,
    arg_out: String,
    cmd_constant: bool,
    cmd_uniform: bool,
    cmd_out_degree: bool,
    cmd_binary_uniform: bool,
    arg_min: Option<f64>,
    arg_max: Option<f64>,
    flag_linear: bool,
    flag_log: bool,
    arg_on_fraction: Option<f64>,
}

fn main() {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    let g = capngraph::load_graph(&args.arg_graph).unwrap();
    let mut out = File::create(&args.arg_out).unwrap();
    if args.cmd_constant {
        bincode::serialize_into(&mut out, &vec![1f64; g.node_count()], bincode::Infinite).unwrap();
    } else if args.cmd_uniform {
        use statrs::distribution::Uniform;
        use rand::distributions::IndependentSample;
        let dist = Uniform::new(args.arg_min.unwrap(), args.arg_max.unwrap()).unwrap();
        let mut rng = rand::thread_rng();
        bincode::serialize_into(&mut out,
                                &(0..g.node_count())
                                    .map(|_| dist.ind_sample(&mut rng))
                                    .collect::<Vec<f64>>(),
                                bincode::Infinite)
            .unwrap();
    } else if args.cmd_out_degree {
        let scale = |w: f64| {
            let w = g.node_count() as f64 / g.edge_count() as f64 * w;
            if args.flag_linear { w } else { w.ln() }
        };
        bincode::serialize_into(&mut out,
                                &g.node_indices()
                                    .map(|v| g.neighbors_directed(v, Outgoing).count() as f64)
                                    .map(scale)
                                    .collect::<Vec<f64>>(),
                                bincode::Infinite)
            .unwrap();
    } else if args.cmd_binary_uniform {
        use rand::distributions::IndependentSample;
        use std::collections::BTreeSet;
        let mut rng = rand::thread_rng();
        let targets = rand::sample(&mut rng,
                                   g.node_indices(),
                                   (g.node_count() as f64 * args.arg_on_fraction.unwrap())
                                       .ceil() as usize);
        let targets = targets.into_iter().collect::<BTreeSet<NodeIndex>>();
        let dist = statrs::distribution::Uniform::new(args.arg_min.unwrap(), args.arg_max.unwrap())
            .unwrap();

        bincode::serialize_into(&mut out,
                                &g.node_indices()
                                    .map(|v| if targets.contains(&v) {
                                        dist.ind_sample(&mut rng)
                                    } else {
                                        0.0
                                    })
                                    .collect::<Vec<f64>>(),
                                bincode::Infinite)
            .unwrap();
    }
}
