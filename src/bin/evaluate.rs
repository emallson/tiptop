extern crate rayon;
extern crate rand;
extern crate statrs;
extern crate petgraph;
extern crate vec_graph;
extern crate capngraph;
extern crate ris;
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
#[macro_use]
extern crate nom;

use std::iter::FromIterator;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use petgraph::visit::NodeCount;
use vec_graph::{Graph, NodeIndex};
use rayon::prelude::*;
use slog::{Logger, DrainExt};
use serde_json::to_string as json_string;
use statrs::distribution::Categorical;
use rand::Rng;
use rand::distributions::IndependentSample;
use rand_mersenne_twister::{MTRng64, mersenne};
use bincode::{deserialize_from as bin_read_from, Infinite};
use std::cell::RefCell;
use std::path::Path;

use ris::*;

#[cfg_attr(rustfmt, rustfmt_skip)]
const USAGE: &'static str = "
Estimate the influence of a seed set via reverse influence sampling.

If <delta> is not given, 1/n is used as a default.

If --benefits are not given, then they are treated as uniformly 1.
Thus, ommitting both is equivalent to the normal unweighted IM problem.

Usage:
    evaluate <graph> <model> <seeds> <epsilon> [<delta>] [options]
    evaluate (-h | --help)

Options:
    -h --help              Show this screen.
    --log <logfile>        Log to given file.
    --threads <threads>    Number of threads to use.
    --benefits <ben-file>  Node benefits. Generated via the `build-data` binary.
";

#[derive(Serialize, Deserialize, Debug)]
struct Args {
    arg_graph: String,
    arg_model: Model,
    arg_seeds: String,
    arg_epsilon: f64,
    arg_delta: Option<f64>,
    flag_log: Option<String>,
    flag_threads: Option<usize>,
    flag_costs: Option<String>,
    flag_benefits: Option<String>,
}

type BenVec = Vec<f64>;
type BenDist = Categorical;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum Model {
    IC,
    LT,
}

named!(seed<NodeIndex>, map!(map_res!(map_res!(ws!(nom::digit), std::str::from_utf8), std::str::FromStr::from_str), NodeIndex::new));
named!(seeds<BTreeSet<NodeIndex>>, terminated!(map!(many1!(seed), BTreeSet::from_iter), eof!()));

fn load_seeds<P: AsRef<Path>>(path: P) -> BTreeSet<NodeIndex> {
    let input = File::open(path)
        .expect("Unable to open seed file")
        .bytes()
        .collect::<Result<Vec<u8>, _>>()
        .expect("Unable to read seed file");
    let res = seeds(&input);
    res.to_full_result().expect("Unable to parse seed file")
}

thread_local!(static RNG: RefCell<MTRng64> = RefCell::new(mersenne()));

/// Construct a reverse-reachable sample according to the BSA algorithm (see the SSA paper) under
/// either the IC or LT model.
///
/// If no benefits are given, this does uniform sampling.
///
/// WARN: copied from main.rs -- need to split into lib and unify
fn rr_sample<R: Rng>(rng: &mut R,
                     g: &Graph<(), f32>,
                     model: Model,
                     weights: &Option<BenDist>)
                     -> BTreeSet<NodeIndex> {
    if let &Some(ref dist) = weights {
        let v = dist.ind_sample(rng);
        assert_eq!(v, v.trunc());
        let v = NodeIndex::new(v as usize);
        match model {
            Model::IC => IC::new(rng, g, v),
            Model::LT => LT::new(rng, g, v),
        }
    } else {
        match model {
            Model::IC => IC::new_uniform_with(rng, g),
            Model::LT => LT::new_uniform_with(rng, g),
        }
    }
}

/// Constructs an (ε, δ)-approximation of the CTVM objective value.
fn evaluate(g: &Graph<(), f32>,
            bens: Option<BenVec>,
            model: Model,
            seeds: &BTreeSet<NodeIndex>,
            epsilon: f64,
            delta: f64,
            log: Logger)
            -> f64 {
    // for the sake of efficiency, we compute batches of size `step`
    const STEP: usize = 10_000;

    let mut num_cov: usize = 0;
    let mut num_sets: usize = 0;
    let n = g.node_count() as f64;
    let dist = bens.as_ref().map(|w| Categorical::new(w).unwrap());
    let dr = &dist;

    let eps2 = epsilon / 16.0;
    let delta2 = delta / 16.0;
    let eps2p = eps2 / (1.0 - eps2);
    let delta2p = delta2;
    let lam2 = 1.0 +
               (2.0 + 2.0 / 3.0 * eps2p) * (1.0 + eps2p) * (1.0 / delta2p).ln() * eps2p.powi(-2);
    let lam2 = lam2.ceil() as usize;

    while num_cov < lam2 {
        // info!(log, "boosting coverage"; "cov" => num_cov, "λ₂" => lam2);
        num_cov += (0..STEP)
            .into_par_iter()
            .map(move |_| RNG.with(|r| rr_sample(&mut *r.borrow_mut(), g, model, dr)))
            .map(|rr| rr.intersection(seeds).take(1).count())
            .sum::<usize>();
        num_sets += STEP;
        info!(log, "adding samples..."; "Λ₂" => lam2, "covered" => num_cov, "samples generated" => num_sets);
    }
    n * (num_cov as f64 / num_sets as f64)
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
    let g = Graph::oriented_from_edges(capngraph::load_edges(args.arg_graph.as_str()).unwrap(),
                                       petgraph::Incoming);
    let delta = args.arg_delta.unwrap_or(1.0 / g.node_count() as f64);
    let bens: Option<BenVec> = args.flag_benefits
        .as_ref()
        .map(|path| bin_read_from(&mut File::open(path).unwrap(), Infinite).unwrap());

    if let Some(ref b) = bens {
        assert_eq!(b.len(), g.node_count());
    }

    let seeds = load_seeds(&args.arg_seeds);

    let influence = evaluate(&g,
                             bens,
                             args.arg_model,
                             &seeds,
                             args.arg_epsilon,
                             delta,
                             log.new(o!("section" => "verify")));
    info!(log, "estimated influence"; "seeds" => json_string(&seeds.into_iter().map(|node| node.index()).collect::<Vec<_>>()).unwrap(), "influence" => influence);
}
