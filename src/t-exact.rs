extern crate rayon;
extern crate rand;
extern crate statrs;
extern crate petgraph;
extern crate vec_graph;
extern crate gurobi;
extern crate capngraph;
extern crate bit_set;
extern crate docopt;
#[macro_use]
extern crate slog;
extern crate slog_stream;
extern crate slog_term;
extern crate slog_json;
extern crate serde;
extern crate serde_json;
extern crate bincode;
#[macro_use]
extern crate serde_derive;
extern crate rand_mersenne_twister;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

use std::fs::File;
use std::cell::RefCell;

use petgraph::prelude::*;
use rayon::prelude::*;

use slog::{Logger, DrainExt};
use statrs::distribution::Uniform;
use rand::Rng;
use rand::distributions::IndependentSample;
use bit_set::BitSet;
use rand_mersenne_twister::{MTRng64, mersenne};
use serde_json::to_string as json_string;
use std::iter::once;
use bincode::{Infinite, deserialize_from as bin_read_from};

#[cfg_attr(rustfmt, rustfmt_skip)]
const USAGE: &str = "
Run the T-EXACT IP. Be warned that THIS IS VERY SLOW. Only run on small datasets.

Only the IC model is supported.

See https://arxiv.org/abs/1701.08462 for exact definition of this IP.

Usage:
    t-exact <graph> IC <k> <samples> [options]
    t-exact (-h | --help)

Options:
    -h --help               Show this screen.
    --log <logfile>         Log to given file.
    --threads <threads>     Number of threads to use.
    --costs <path>          Path to costs file.
    --benefits <path>       Path to benefits file.
";

#[derive(Debug, Serialize, Deserialize)]
struct Args {
    arg_graph: String,
    arg_k: usize,
    arg_samples: usize,
    flag_log: Option<String>,
    flag_threads: Option<usize>,
    flag_costs: Option<String>,
    flag_benefits: Option<String>,
}

thread_local!(static RNG: RefCell<MTRng64> = RefCell::new(mersenne()));

type CostVec = Vec<f64>;
type BenVec = Vec<f64>;
type InfGraph = Graph<(), f32>;
type Sample = BitSet;


fn sample<R: Rng>(rng: &mut R, g: &InfGraph) -> Sample {
    let uniform = Uniform::new(0.0, 1.0).expect("Unable to construct uniform dist (?!?!?)");
    let mut sample = Sample::default();
    for edge in g.edge_references() {
        if uniform.ind_sample(rng) < *edge.weight() as f64 {
            sample.insert(edge.id().index());
        }
    }

    sample
}

fn ip(g: &InfGraph,
      samples: &Vec<Sample>,
      k: usize,
      benefits: Option<BenVec>,
      costs: Option<CostVec>,
      threads: usize,
      log: Logger)
      -> Result<Vec<usize>, String> {
    use gurobi::*;
    let mut env = Env::new();
    env.set_threads(threads)?;
    let mut model = Model::new(&env)?;
    model.set_objective_type(ObjectiveType::Maximize)?;

    #[allow(non_snake_case)]
    let T = samples.len() as f64;

    info!(log, "building IP");
    let s = g.node_indices()
        .map(|_| model.add_var(0.0, VariableType::Binary).unwrap())
        .collect::<Vec<_>>();
    let inv = g.node_indices().collect::<Vec<_>>();

    model.add_con(costs.map(|c| Constraint::build().weighted_sum(&s, c).is_less_than(k as f64))
            .unwrap_or_else(|| Constraint::build().sum(&s).is_less_than(k as f64)))?;
    for sample in samples {
        let x = g.node_indices()
            .map(|u| {
                model.add_var(benefits.as_ref().map(|b| b[u.index()]).unwrap_or(1.0) / T,
                             VariableType::Binary)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        for node in g.node_indices() {
            // for each node to be active, either (a) it must be seeded, or (b) it must have an
            // active incoming neighbor in the sample
            let con = Constraint::build()
                .sum(g.edges_directed(node, Incoming)
                    .filter(|e| sample.contains(e.id().index()))
                    .map(|e| x[e.source().index()])
                    .chain(once(s[node.index()])))
                .plus(x[node.index()], -1.0)
                .is_greater_than(0.0);
            model.add_con(con)?;
        }
    }

    let sol = model.optimize()?;
    info!(log, "found solution"; "value" => sol.value()?);
    Ok(sol.variables(s[0], s[s.len() - 1])?
        .iter()
        .enumerate()
        .filter_map(|(i, &f)| if f == 1.0 { Some(inv[i].index()) } else { None })
        .collect())
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    if let Some(threads) = args.flag_threads {
        rayon::initialize(rayon::Configuration::new().num_threads(threads)).unwrap();
    }

    let log =
        match args.flag_log {
            Some(ref filename) => slog::Logger::root(slog::Duplicate::new(slog_term::streamer().color().compact().build(),
            slog_stream::stream(File::create(filename).unwrap(), slog_json::default())).fuse(), o!("version" => env!("CARGO_PKG_VERSION"))),
            None => {
                slog::Logger::root(slog_term::streamer().color().compact().build().fuse(),
                                   o!("version" => env!("CARGO_PKG_VERSION")))
            }
        };

    info!(log, "parameters"; "args" => json_string(&args).unwrap());
    info!(log, "loading graph"; "path" => args.arg_graph);
    let g = capngraph::load_graph(args.arg_graph.as_str()).expect("Unable to load graph.");
    let costs: Option<CostVec> = args.flag_costs
        .as_ref()
        .map(|path| bin_read_from(&mut File::open(path).unwrap(), Infinite).unwrap());
    let bens: Option<BenVec> = args.flag_benefits
        .as_ref()
        .map(|path| bin_read_from(&mut File::open(path).unwrap(), Infinite).unwrap());

    let samples = (0..args.arg_samples)
        .into_par_iter()
        .map(|_| RNG.with(|r| sample(&mut *r.borrow_mut(), &g)))
        .collect::<Vec<_>>();

    let soln = ip(&g,
                  &samples,
                  args.arg_k,
                  bens,
                  costs,
                  args.flag_threads.unwrap_or(1),
                  log.new(o!("section" => "ip")))
        .unwrap();

    info!(log, "optimal solution"; "seeds" => json_string(&soln).unwrap());
}
