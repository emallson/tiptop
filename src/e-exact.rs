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

use petgraph::visit::EdgeFiltered;
use petgraph::algo::bellman_ford;
use slog::{Logger, DrainExt};
use statrs::distribution::Uniform;
use rand::Rng;
use rand::distributions::IndependentSample;
use bit_set::BitSet;
use rand_mersenne_twister::{MTRng64, mersenne};
use serde_json::to_string as json_string;
use bincode::{Infinite, deserialize_from as bin_read_from};

#[cfg_attr(rustfmt, rustfmt_skip)]
const USAGE: &str = "
Run the E-EXACT IP. Be warned that THIS IS VERY SLOW. Only run on small datasets.

Only the IC model is supported.

See https://arxiv.org/abs/1701.08462 for exact definition of this IP.

Usage:
    e-exact <graph> IC <k> <samples> [options]
    e-exact (-h | --help)

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
type Cycle = Vec<EdgeIndex>;
type CycleVec = Vec<Cycle>;


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
      sample_cycles: &Vec<CycleVec>,
      k: usize,
      benefits: Option<&BenVec>,
      costs: Option<&CostVec>,
      threads: usize,
      log: Logger)
      -> Result<(Vec<usize>, Vec<Vec<bool>>), String> {
    use gurobi::*;
    let mut env = Env::new();
    env.set_threads(threads)?;
    let mut model = Model::new(&env)?;
    model.set_objective_type(ObjectiveType::Maximize)?;

    #[allow(non_snake_case)]
    let T = samples.len() as f64;

    info!(log, "building IP");
    let s = g.node_indices()
        .map(|u| {
            model.add_var(benefits.as_ref().map(|b| b[u.index()]).unwrap_or(1.0),
                         VariableType::Binary)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let inv = g.node_indices().collect::<Vec<_>>();

    // cardinality constraint
    model.add_con(costs.map(|c| Constraint::build().weighted_sum(&s, c).is_less_than(k as f64))
            .unwrap_or_else(|| Constraint::build().sum(&s).is_less_than(k as f64)))?;
    let mut ys = Vec::with_capacity(samples.len());
    for (sample, cycles) in samples.iter().zip(sample_cycles) {
        let y = g.edge_references()
            .map(|e| {
                model.add_var(benefits.as_ref().map(|b| b[e.target().index()]).unwrap_or(1.0) / T,
                             VariableType::Binary)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        debug!(log, "adding edge connectivity constraints");
        for eref in g.edge_references() {
            let con = Constraint::build()
                .sum(g.edges_directed(eref.source(), Incoming)
                    .filter(|e| sample.contains(e.id().index()))
                    .map(|e| y[e.id().index()]))
                .plus(s[eref.source().index()], 1.0)
                .plus(y[eref.id().index()], -1.0)
                .is_greater_than(0.0);
            model.add_con(con)?;
        }

        debug!(log, "adding node connectivity constraints");
        for v in g.node_indices() {
            let con = Constraint::build()
                .sum(g.edges_directed(v, Incoming)
                    .filter(|e| sample.contains(e.id().index()))
                    .map(|e| y[e.id().index()]))
                .plus(s[v.index()], 1.0)
                .is_less_than(1.0);
            model.add_con(con)?;
        }

        debug!(log, "adding {} cycle constraints", cycles.len());
        for cycle in cycles {
            let con = Constraint::build()
                .sum(cycle.iter().map(|e| y[e.index()]))
                .is_less_than(cycle.len() as f64 - 1.0);
            model.add_con(con)?;
        }
        ys.push(y);
    }

    let sol = model.optimize()?;
    info!(log, "found solution"; "value" => sol.value()?);
    Ok((sol.variables(s[0], s[s.len() - 1])?
            .iter()
            .enumerate()
            .filter_map(|(i, &f)| if f == 1.0 { Some(inv[i].index()) } else { None })
            .collect(),
        ys.into_iter()
            .map(|y| {
                sol.variables(y[0], y[y.len() - 1]).unwrap().iter().map(|&v| v == 1.0).collect()
            })
            .collect()))
}

fn e_exact(g: &InfGraph,
           samples: &Vec<Sample>,
           k: usize,
           benefits: Option<BenVec>,
           costs: Option<CostVec>,
           threads: usize,
           log: Logger)
           -> Result<Vec<usize>, String> {
    let cycles = vec![vec![]; samples.len()];
    loop {
        let (soln, active_edges) = ip(&g,
                                      &samples,
                                      &cycles,
                                      k,
                                      benefits.as_ref(),
                                      costs.as_ref(),
                                      threads,
                                      log.new(o!("section" => "ip")))?;
        let mut added_cycles = false;
        'cycle: for active in active_edges {
            let gp = EdgeFiltered::from_fn(&g, |e| active[e.id().index()]);
            for u in g.node_indices() {
                let (weights, _) = bellman_ford(&gp, u).unwrap();
                for v in g.edges_directed(u, Incoming)
                    .filter(|e| active[e.id().index()])
                    .map(|e| e.source()) {
                    if weights[v.index()].is_finite() {
                        panic!("cycle found! {:?} {} -- cycle handling is NOT implemented since I haven't yet needed it", v, weights[v.index()]);
                    }
                }
            }
        }
        if !added_cycles {
            return Ok(soln);
        }
    }
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

    let soln = e_exact(&g,
                       &samples,
                       args.arg_k,
                       bens,
                       costs,
                       args.flag_threads.unwrap_or(1),
                       log.new(o!("section" => "e-exact")))
        .unwrap();

    info!(log, "optimal solution"; "seeds" => json_string(&soln).unwrap());
}
