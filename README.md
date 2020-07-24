# rust-reversi
Reversi solver on Rust

## License

GPLv3

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
$ cargo run --release
```

## Benchmark result

Date: 2020/07/24
Hardware: AMD Ryzen 9 3900X, DDR4-3200 64GB
Environment: Linux 5.4.0-42, Ubuntu 20.04, rustc 1.46.0-nightly

FFO 40-59

|  No|empties|score|nodes|time|
|----|----|----|-------|-------|
|  40|  20| +38|    54M|  0.31s|
|  41|  22|  +0|   163M|  0.63s|
|  42|  22|  +6|    97M|  0.43s|
|  43|  23| -12|   469M|  1.80s|
|  44|  23| -14|   510M|  1.52s|
|  45|  24|  +6|  4510M| 11.72s|
|  46|  24|  -8|   667M|  2.14s|
|  47|  25|  +4|   146M|  0.95s|
|  48|  25| +28|  2131M|  5.63s|
|  49|  26| +16|  3720M| 11.88s|
|  50|  26| +10|  9107M| 29.77s|
|  51|  27|  +6|  4758M| 18.22s|
|  52|  27|  +0|  3761M| 14.48s|
|  53|  28|  -2| 39255M|125.12s|
|  54|  28|  -2| 30701M| 91.79s|
|  55|  29|  +0|175548M|491.34s|
|  56|  29|  +2|  6079M| 32.52s|
|  57|  30| -10| 31188M|148.35s|
|  58|  30|  +4| 13842M| 76.12s|
|  59|  34| +64|  8029M|119.61s|
