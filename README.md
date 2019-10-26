# rust-reversi
Reversi solver on Rust

## prerequirements

AVX2 and BMI2 enabled x86\_64 CPU (Intel: Haswell or lator, AMD: Excavator or lator)

## Setup

prepare evaluation table

```Shell
$ wget https://poyo.me/reversi/table191027.tar.gz
$ tar xf table191027.tar.gz
```

## Run

compile and run

```Shell
$ RUSTFLAGS="-C target-cpu=native" cargo run --release
```
