extern crate rayon;
extern crate rand;
extern crate petgraph;
#[macro_use]
extern crate rplex;
extern crate capngraph;
extern crate ris;
extern crate bit_set;
extern crate docopt;
extern crate rustc_serialize;
#[macro_use]
extern crate slog;
extern crate slog_stream;
extern crate slog_term;
extern crate slog_json;
extern crate serde_json;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use petgraph::prelude::*;
use rayon::prelude::*;
use rplex::*;
use slog::{Logger, DrainExt};
use serde_json::to_string as json_string;

use ris::*;

#[cfg_attr(rustfmt, rustfmt_skip)]
const USAGE: &'static str = "
Run the TipTop algorithm.

Usage:
    tiptop <graph> <model> <k> <epsilon> [<delta>] [--log <logfile>]
    tiptop (-h | --help)

Options:
    -h --help           Show this screen.
    --log <logfile>     Log to given file.
";

#[derive(Debug, RustcDecodable)]
struct Args {
    arg_graph: String,
    arg_model: Model,
    arg_k: usize,
    arg_epsilon: f64,
    arg_delta: Option<f64>,
    flag_log: Option<String>,
}

#[derive(Clone, Copy, Debug, RustcDecodable)]
enum Model {
    IC,
    LT,
}

fn binom(n: usize, k: usize) -> f64 {
    (1..k + 1).map(|i| (n + 1 - i) as f64 / i as f64).fold(1.0, |p, x| p * x)
}

fn ilp_mc(g: &Graph<(), f32>,
          rr_sets: &Vec<BTreeSet<NodeIndex>>,
          k: usize,
          log: &Logger)
          -> BTreeSet<NodeIndex> {
    let mut env = Env::new().unwrap();
    env.set_param(EnvParam::ScreenOutput(false)).unwrap();
    let mut prob = Problem::new(&env, "ilp_mc").unwrap();

    let nodes = g.node_indices().enumerate().map(|(i, node)| (node, i)).collect::<BTreeMap<_, _>>();
    let inv = g.node_indices().collect::<Vec<_>>();
    let s = g.node_indices()
        .map(|node| {
            prob.add_variable(var!((format!("s{}", node.index())) -> 0.0 as Binary)).unwrap()
        })
        .collect::<Vec<_>>();

    #[allow(unused_parens)]
    prob.add_constraint(con!("cardinality": (k as f64) >= sum (s.iter()))).unwrap();

    for (i, set) in rr_sets.iter().enumerate() {
        let y = prob.add_variable(var!((format!("y{}", i)) -> 1.0 as Binary)).unwrap();
        let els = set.iter().map(|node| s[nodes[node]]).collect::<Vec<_>>();
        #[allow(unused_parens)]
        let mut con = con!((format!("rr{}", i)): 1.0 <= sum (els.iter()));
        con.add_wvar(WeightedVariable::new_idx(y, 1.0));
        prob.add_constraint(con).unwrap();
    }

    // TODO: confirm the switch from max to min
    prob.set_objective_type(ObjectiveType::Minimize).unwrap();

    let Solution { variables, .. } = prob.solve().unwrap();

    s.iter()
        .filter_map(|&i| match variables[i] {
            VariableValue::Binary(b) => if b { Some(inv[i]) } else { None },
            _ => None,
        })
        .collect()
}

fn verify(g: &Graph<(), f32>,
          seeds: &BTreeSet<NodeIndex>,
          model: Model,
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

    let step = 10_000;

    for i in 0..v_max - 1 {
        eps2 = eps.min(1.0) / 2f64.powi(i as i32);
        let eps2p = eps2 / (1.0 - eps2);
        let delta2p = delta2 / (v_max as f64 * t_max as f64);
        let lam2 =
            1.0 + (2.0 + 2.0 / 3.0 * eps2p) * (1.0 + eps2p) * (2.0 / delta2p).ln() * eps2p.powi(-2);

        info!(log, "boosting coverage"; "cov" => num_cov, "λ₂" => lam2);
        while num_cov < lam2 {
            let mut next_sets = Vec::with_capacity(step);
            (0..step)
                .into_par_iter()
                .map(|_| match model {
                    Model::IC => ris::IC::new_uniform(&g),
                    Model::LT => ris::LT::new_uniform(&g),
                })
                .collect_into(&mut next_sets);

            num_cov += cov(&next_sets, &seeds);
            rr_sets.append(&mut next_sets);
            if rr_sets.len() > tcap {
                info!(log, "T_cap exceeded"; "tcap" => tcap);
                return (false, eps1, 2.0 * eps2);
            }
        }

        let gamma = g.node_indices().count() as f64; // all benefits are 1
        let b_ver = gamma * num_cov / rr_sets.len() as f64;
        eps1 = 1.0 - b_ver / b_r;

        if eps1 > eps {
            info!(log, "eps1 > eps"; "eps1" => eps1, "eps" => eps);
            return (false, eps1, eps2);
        }

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

fn cov(rr_sets: &Vec<BTreeSet<NodeIndex>>, seeds: &BTreeSet<NodeIndex>) -> f64 {
    rr_sets.iter().map(|rr| rr.intersection(seeds).take(1).count() as f64).sum::<f64>()
}

fn tiptop(g: Graph<(), f32>,
          model: Model,
          k: usize,
          eps: f64,
          delta: f64,
          log: Logger)
          -> BTreeSet<NodeIndex> {
    let n: f64 = g.node_count() as f64;
    let lambda = (1.0 + eps) * (2.0 + 2.0 / 3.0 * eps) * eps.powi(-2) * (2.0 / delta).ln();
    let mut t = 1.0;

    let t_max = (2.0 * (n / eps).ln()).ceil();
    let v_max: u32 = 6;
    let lam_max = (1.0 + eps) * (2.0 + 2.0 / 3.0 * eps) * 2.0 * eps.powi(-2) *
                  ((2.0 / (delta / 4.0)).ln() + binom(n as usize, k).ln());

    let mut rr_sets: Vec<BTreeSet<NodeIndex>> = Vec::new();
    loop {
        let nt = (lambda * (eps * t).exp()) as usize;
        info!(log, "sampling more sets"; "total" => nt, "additional" => nt - rr_sets.len());
        let mut next_sets = Vec::with_capacity(nt - rr_sets.len());
        (0..nt - rr_sets.len())
            .into_par_iter()
            .map(|_| IC::new_uniform(&g))
            .collect_into(&mut next_sets);
        rr_sets.append(&mut next_sets);

        info!(log, "solving ip");
        let seeds = ilp_mc(&g, &rr_sets, k, &log);
        info!(log, "verifying solution");
        let (passed, eps_1, eps_2) = verify(&g,
                                            &seeds,
                                            model,
                                            eps,
                                            delta,
                                            cov(&rr_sets, &seeds) * n / rr_sets.len() as f64,
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
        let dt_max = (2.0 / eps).ceil();
        t += dt_max.min(1f64.max(((1.0 / eps) * (eps_1 / eps_2).powi(2).ln()).ceil()));
    }
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.decode())
        .unwrap_or_else(|e| e.exit());

    let log =
        match args.flag_log {
            Some(filename) => slog::Logger::root(slog::Duplicate::new(slog_term::streamer().color().compact().build(),
                                                                  slog_stream::stream(File::create(filename).unwrap(), slog_json::default())).fuse(), o!("version" => env!("CARGO_PKG_VERSION"))),
            None => {
                slog::Logger::root(slog_term::streamer().color().compact().build().fuse(),
                                   o!("version" => env!("CARGO_PKG_VERSION")))
            }
        };

    let g = capngraph::load_graph(args.arg_graph.as_str()).unwrap();
    let delta = args.arg_delta.unwrap_or(1.0 / g.node_count() as f64);
    let seeds = tiptop(g,
                       args.arg_model,
                       args.arg_k,
                       args.arg_epsilon,
                       delta,
                       log.new(o!("section" => "tiptop")));
    info!(log, "optimal solution"; "seeds" => json_string(&seeds.into_iter().map(|node| node.index()).collect::<Vec<_>>()).unwrap());
}
