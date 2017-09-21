extern crate rayon;
extern crate rand;
extern crate statrs;
extern crate petgraph;
extern crate vec_graph;
#[macro_use]
extern crate rplex;
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

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use petgraph::visit::NodeCount;
use vec_graph::{Graph, NodeIndex, EdgeIndex};
use rayon::prelude::*;
use rplex::*;
use slog::{Logger, DrainExt};
use serde_json::to_string as json_string;
use statrs::distribution::Categorical;
use rand::Rng;
use rand::distributions::IndependentSample;
use rand_mersenne_twister::{MTRng64, mersenne};
use bincode::{deserialize_from as bin_read_from, Infinite};
use std::cell::RefCell;

use ris::*;

#[cfg_attr(rustfmt, rustfmt_skip)]
const USAGE: &'static str = "
Run the TipTop algorithm.

If <delta> is not given, 1/n is used as a default.

If --costs are not given, then they are treated as uniformly 1.
If --benefits are not given, then they are treated as uniformly 1.
Thus, ommitting both is equivalent to the normal unweighted IM problem.

See https://arxiv.org/abs/1701.08462 for the latest version of the paper.

Usage:
    tiptop <graph> <model> <k> <epsilon> [<delta>] [options]
    tiptop (-h | --help)

Options:
    -h --help              Show this screen.
    --log <logfile>        Log to given file.
    --threads <threads>    Number of threads to use.
    --costs <cost-file>    Node costs. Generated via the `build-data` binary.
    --benefits <ben-file>  Node benefits. Generated via the `build-data` binary.

    --cov-jump <factor>    Check that Cov_R(S_k) ≥ Λ before attempting verification.
                           This skips many early stages of verification where verify() 
                           takes longer than solving the IP. While this is in effect,
                           an additional <factor> samples are generated each round.
                           After this condition is satisfied, IncreaseSamples() is used.
                           Recommended <factor> is 0.2.
";

#[derive(Debug, Serialize, Deserialize)]
struct Args {
    arg_graph: String,
    arg_model: Model,
    arg_k: usize,
    arg_epsilon: f64,
    arg_delta: Option<f64>,
    flag_log: Option<String>,
    flag_threads: Option<usize>,
    flag_costs: Option<String>,
    flag_benefits: Option<String>,
    flag_cov_jump: Option<f64>,
}

type CostVec = Vec<f64>;
type BenVec = Vec<f64>;
type BenDist = Categorical;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum Model {
    IC,
    LT,
}

thread_local!(static RNG: RefCell<MTRng64> = RefCell::new(mersenne()));

/// log(n choose k) computed using the sum form to avoid overflow.
fn logbinom(n: usize, k: usize) -> f64 {
    (1..(k + 1)).map(|i| ((n + 1 - i) as f64).ln() - (i as f64).ln()).sum()
}

/// Constructs and solves the IP given in eqn. (16)-(19) in the paper.
///
/// TODO: re-use prior solutions as a starting point, re-use previous construction (only adding new
/// variables as needed, never removing as they are never removed).
fn ilp_mc(g: &Graph<(), f32>,
          rr_sets: &Vec<BTreeSet<NodeIndex>>,
          costs: &Option<CostVec>,
          k: usize,
          threads: usize,
          log: &Logger)
          -> BTreeSet<NodeIndex> {
    let mut env = Env::new().unwrap();
    env.set_param(EnvParam::ScreenOutput(false)).unwrap();
    env.set_param(EnvParam::Threads(threads as u64)).unwrap();
    env.set_param(EnvParam::ParallelDeterministic(false)).unwrap();
    let mut prob = Problem::new(&env, "ilp_mc").unwrap();

    let nodes = g.node_indices().enumerate().map(|(i, node)| (node, i)).collect::<BTreeMap<_, _>>();
    let inv = g.node_indices().collect::<Vec<_>>();
    let s = g.node_indices()
        .map(|node| {
            prob.add_variable(var!((format!("s{}", node.index())) -> 0.0 as Binary)).unwrap()
        })
        .collect::<Vec<_>>();

    // weights the seed variables by their cost (or 1.0 if no costs are given)
    let weighted_s: Vec<_> =
        costs.as_ref().map_or_else(|| s.iter().map(|&si| (si, 1.0)).collect(), |costs| {
            s.iter().zip(costs).map(|(&si, &c)| (si, c as f64)).collect()
        });
    // constraint (17)
    #[allow(unused_parens)]
    prob.add_constraint(con!("cardinality": (k as f64) >= wsum (&weighted_s))).unwrap();

    for (i, set) in rr_sets.iter().enumerate() {
        let y = prob.add_variable(var!((format!("y{}", i)) -> 1.0 as Binary)).unwrap();
        let els = set.iter().map(|node| s[nodes[node]]).collect::<Vec<_>>();
        // constraint (18)
        #[allow(unused_parens)]
        let con = con!((format!("rr{}", i)): 1.0 <= 1.0 y + sum (&els));
        prob.add_constraint(con).unwrap();
    }

    prob.set_objective_type(ObjectiveType::Minimize).unwrap();

    let Solution { variables, .. } = prob.solve().unwrap();

    s.iter()
        .filter_map(|&i| match variables[i] {
            VariableValue::Binary(b) => if b { Some(inv[i]) } else { None },
            _ => None,
        })
        .collect()
}

/// This function should *almost* exactly match Algorithm 2 given in
/// [arXiv](http://arxiv.org/abs/1701.08462). The only difference is that we compute RR samples in
/// batches and then process the samples in the batch in sequence. This makes much better use of
/// multiprocessing.
fn verify(g: &Graph<(), f32>,
          seeds: &BTreeSet<NodeIndex>,
          model: Model,
          gamma: f64,
          dist: &Option<BenDist>,
          eps: f64,
          delta: f64,
          b_r: f64,
          v_max: usize,
          t_max: usize,
          tcap: usize,
          log: Logger)
          -> (bool, f64, f64) {
    let mut rr_sets = Vec::new();
    let delta2 = delta / 4.0;
    let mut num_cov = 0.0;
    let mut eps1 = std::f64::INFINITY;
    let mut eps2 = std::f64::INFINITY;

    // for the sake of efficiency, we compute batches of size `step`
    let step = 10_000;

    for i in 0..v_max - 1 {
        eps2 = eps.min(1.0) / 2f64.powi(i as i32);
        let eps2p = eps2 / (1.0 - eps2);
        let delta2p = delta2 / (v_max as f64 * t_max as f64);
        let lam2 =
            1.0 + (2.0 + 2.0 / 3.0 * eps2p) * (1.0 + eps2p) * (2.0 / delta2p).ln() * eps2p.powi(-2);

        while num_cov < lam2 {
            // info!(log, "boosting coverage"; "cov" => num_cov, "λ₂" => lam2);
            let mut next_sets = Vec::with_capacity(step);
            (0..step)
                .into_par_iter()
                .map(move |_| RNG.with(|r| rr_sample(&mut *r.borrow_mut(), &g, model, dist)))
                .collect_into(&mut next_sets);

            num_cov += cov(&next_sets, &seeds);
            rr_sets.append(&mut next_sets);
            if rr_sets.len() > tcap {
                info!(log, "T_cap exceeded"; "tcap" => tcap);
                return (false, eps1, 2.0 * eps2);
            }
        }

        let b_ver = gamma * num_cov / rr_sets.len() as f64;
        eps1 = 1.0 - b_ver / b_r;

        if eps1 > eps {
            info!(log, "eps1 > eps"; "eps1" => eps1, "eps" => eps);
            return (false, eps1, eps2);
        }

        // NOTE: the use of the (undefined) \delta_1 is a typo. it should be \delta_2. Working on
        // getting this fixed.
        let eps3 = ((3.0 * (t_max as f64 / delta2).ln()) / ((1.0 - eps1) * (1.0 - eps2) * num_cov))
            .sqrt();

        if (1.0 - eps1) * (1.0 - eps2) * (1.0 - eps3) > (1.0 - eps) {
            info!(log, "verification succeeded");
            return (true, eps1, eps2);
        }
    }

    info!(log, "iteration limit exceeded");
    return (false, eps1, eps2);
}

/// For a set of `q` RIS samples, returns `q' ≤ q` the number of samples that intersect with the
/// solution. This is in effect applying line 8 of Alg 2. in arXiv to a size-`q` batch of samples.
fn cov(rr_sets: &Vec<BTreeSet<NodeIndex>>, seeds: &BTreeSet<NodeIndex>) -> f64 {
    rr_sets.iter().map(|rr| rr.intersection(seeds).take(1).count() as f64).sum::<f64>()
}

/// Construct a reverse-reachable sample according to the BSA algorithm (see the SSA paper) under
/// either the IC or LT model.
///
/// If no benefits are given, this does uniform sampling.
///
/// TODO: re-use rng object, also MT?
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

/// The core method of the paper. As much as possible, variables and methods are named to match
/// those of the paper to ease verification of the code.
fn tiptop(g: Graph<(), f32>,
          costs: Option<CostVec>,
          benefits: Option<BenVec>,
          model: Model,
          k: usize,
          eps: f64,
          delta: f64,
          threads: usize,
          cov_jump: Option<f64>,
          log: Logger)
          -> BTreeSet<NodeIndex> {
    let mut cov_jump_passed = false;
    let n: f64 = g.node_count() as f64;
    // gamma is either `n` (no benefits) or the sum of benefits
    let gamma: f64 = benefits.as_ref().map_or_else(|| g.node_count() as f64,
                                                   |b| b.iter().map(|&b| b as f64).sum::<f64>());
    let lambda = (1.0 + eps) * (2.0 + 2.0 / 3.0 * eps) * eps.powi(-2) * (2.0 / delta).ln();
    let mut t = 1.0;

    let t_max = (2.0 * (n / eps).ln()).ceil();
    let v_max: u32 = 6;
    let lam_max = (1.0 + eps) * (2.0 + 2.0 / 3.0 * eps) * 2.0 * eps.powi(-2) *
                  ((2.0 / (delta / 4.0)).ln() + logbinom(n as usize, k));
    info!(log, "constants"; "Γ" => gamma, "Λ" => lambda, "Λ_max" => lam_max, "t_max" => t_max, "v_max" => 6);

    let mut rr_sets: Vec<BTreeSet<NodeIndex>> = Vec::new();
    let dist = benefits.as_ref().map(|w| Categorical::new(w).unwrap());
    loop {
        let nt = (lambda * (eps * t).exp()) as usize;
        info!(log, "sampling more sets"; "total" => nt, "additional" => nt - rr_sets.len());
        let mut next_sets = Vec::with_capacity(nt - rr_sets.len());
        (0..nt - rr_sets.len())
            .into_par_iter()
            .map(|_| RNG.with(|rng| rr_sample(&mut *rng.borrow_mut(), &g, model, &dist)))
            .collect_into(&mut next_sets);
        rr_sets.append(&mut next_sets);

        info!(log, "solving ip");
        let seeds = ilp_mc(&g, &rr_sets, &costs, k, threads, &log);
        if !cov_jump_passed {
            // check cov_jump, and possibly skip this round of verification
            if let Some(factor) = cov_jump {
                let covered = cov(&rr_sets, &seeds);
                if covered >= lambda {
                    cov_jump_passed = true;
                    info!(log, "coverage condition passed, proceeding with verification"; "cov" => covered, "Λ" => lambda);
                } else {
                    t += (1.0 + factor).ln() / eps;
                    info!(log, "coverage condition failed, skipping this round of verification"; "cov" => covered, "Λ" => lambda);
                    continue;
                }
            }
        }

        info!(log, "verifying solution");
        let (passed, eps_1, _) = verify(&g,
                                        &seeds,
                                        model,
                                        gamma,
                                        &dist,
                                        eps,
                                        delta,
                                        cov(&rr_sets, &seeds) * gamma / rr_sets.len() as f64,
                                        v_max as usize,
                                        t_max as usize,
                                        2usize.pow(v_max) * nt,
                                        log.new(o!("section" => "verify")));

        if passed {
            info!(log, "solution passed");
            return seeds;
        } else if cov(&rr_sets, &seeds) > lam_max {
            info!(log, "coverage threshold exceeded");
            return seeds;
        }
        // this part corresponds to Alg 3 (IncreaseSamples)
        let dt_max = (2.0 / eps).ceil();
        t += dt_max.min(1f64.max(((1.0 / eps) * (eps_1 / eps).powi(2).ln()).ceil()));
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
            Some(filename) => slog::Logger::root(slog::Duplicate::new(slog_term::streamer().color().compact().build(),
                                                                  slog_stream::stream(File::create(filename).unwrap(), slog_json::default())).fuse(), o!("version" => env!("CARGO_PKG_VERSION"))),
            None => {
                slog::Logger::root(slog_term::streamer().color().compact().build().fuse(),
                                   o!("version" => env!("CARGO_PKG_VERSION")))
            }
        };

    info!(log, "loading graph"; "path" => args.arg_graph);
    let g = Graph::from_edges(capngraph::load_edges(args.arg_graph.as_str()).unwrap());
    let delta = args.arg_delta.unwrap_or(1.0 / g.node_count() as f64);
    let costs: Option<CostVec> = args.flag_costs
        .as_ref()
        .map(|path| bin_read_from(&mut File::open(path).unwrap(), Infinite).unwrap());
    let bens: Option<BenVec> = args.flag_benefits
        .as_ref()
        .map(|path| bin_read_from(&mut File::open(path).unwrap(), Infinite).unwrap());

    if let Some(ref c) = costs {
        assert_eq!(c.len(), g.node_count());
    }

    if let Some(ref b) = bens {
        assert_eq!(b.len(), g.node_count());
    }

    let seeds = tiptop(g,
                       costs,
                       bens,
                       args.arg_model,
                       args.arg_k,
                       args.arg_epsilon,
                       delta,
                       args.flag_threads.unwrap_or(1),
                       args.flag_cov_jump,
                       log.new(o!("section" => "tiptop")));
    info!(log, "optimal solution"; "seeds" => json_string(&seeds.into_iter().map(|node| node.index()).collect::<Vec<_>>()).unwrap());
}

#[cfg(test)]
mod test {
    use rand::thread_rng;
    use rand::distributions::IndependentSample;
    use statrs::distribution::Categorical;

    #[test]
    fn confirm_categorical() {
        // simple test to confirm 100% that categorical distribution works as intended.
        let c = Categorical::new(&[1.0, 2.0, 1.0, 1.0]).unwrap();
        for _ in 0..1000 {
            let i = c.ind_sample(&mut thread_rng());
            println!("{} {} {}", i, i.trunc(), i as usize);
            assert_eq!(i, i.trunc());
            assert_eq!(i.trunc() as usize, i as usize);
        }
    }
}
