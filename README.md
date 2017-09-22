# TipTop

A re-implementation of the TipTop algorithm from [this
paper](https://arxiv.org/abs/1701.08462) in Rust.

Sampling is done using [this library](https://github.com/emallson/ris.rs)
and solving is done with CPLEX via [this
library](https://github.com/emallson/rplex).

*This is not the canonical/reference implementation!*

## Input Format

The graph input is expected to be in
[Capngraph](https://github.com/emallson/capngraph) format. See the linked
repository for conversion tools.

The cost/benefit inputs are constructed using the [included
binary.](./src/bin/build-data.rs)

## License

Obviously, the method is taken from [here](https://arxiv.org/abs/1701.08462).
The code itself is wholly mine at this point, and is made available under the
[BSD 3-Clause License](./LICENSE).
