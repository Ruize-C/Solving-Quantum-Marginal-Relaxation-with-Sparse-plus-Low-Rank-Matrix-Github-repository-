# TFI 2D

This directory contains the data preparation, initialization, main iteration scripts, and a small-system DMRG notebook for ground-state energy estimation of the 2D transverse-field Ising model.

## Files

- `data/fixed_operator.mat`: fixed operator data.
- `script/prepare_data.m`: builds `J`, `R`, `A`, `M`, `Lambda`, and related data, then saves them to `data/<N>_data.mat`.
- `script/init_S0.m`: builds the low-rank plus banded initialization `S0` and saves it to `data/<N>_S0.mat`.
- `script/alg4.m`: runs the main iteration and saves benchmark results to `benchmark_result/<N>_benchmark.mat`.
- `script/dmrg_tfi.ipynb`: ITensors notebook that only estimates the ground-state energy on a small 2D TFI system.

## Run order

1. Run `script/prepare_data.m`
2. Run `script/init_S0.m`
3. Run `script/alg4.m`

## Outputs

- `data/64_data.mat`
- `data/64_S0.mat`
- `benchmark_result/64_benchmark.mat`
