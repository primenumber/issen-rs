# issen-rs

Reversi solver on Rust

## License

GPLv3

## prerequirements

AVX2 and BMI2 enabled x86\_64 CPU (Intel: Haswell or later, AMD: Excavator or later)

## Setup

prepare evaluation table

```Shell
$ wget https://poyo.me/reversi/table220710.tar.gz
$ tar xf table220710.tar.gz
```

## Run

compile and run

```Shell
$ cargo run --release -- ffobench
```

## Benchmark result

- Date: 2023/06/04
- Hardware: AMD Ryzen 9 5950X, DDR4-3200 128GB
- Environment: Linux 5.15.90.1-microsoft-standard-WSL2, Ubuntu 22.04.2 on WSL2, rustc 1.72.0-nightly

FFO 40-59

|No.|empties|result|answer|move|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|:--:|---:|
|40|20|+38|+38|A2|26.6M|   0.129s|205M/s|
|41|22| +0| +0|H4|30.9M|   0.162s|189M/s|
|42|22| +6| +6|G2|68.1M|   0.314s|216M/s|
|43|23|-12|-12|C7|35.7M|   0.265s|134M/s|
|44|23|-14|-14|D2|34.2M|   0.152s|223M/s|
|45|24| +6| +6|B2| 980M|   2.593s|378M/s|
|46|24| -8| -8|B3| 154M|   0.752s|205M/s|
|47|25| +4| +4|G2|35.0M|   0.257s|135M/s|
|48|25|+28|+28|F6| 536M|   1.443s|371M/s|
|49|26|+16|+16|E1|1.95G|   7.755s|251M/s|
|50|26|+10|+10|D8|2.52G|   5.911s|426M/s|
|51|27| +6| +6|A3| 661M|   3.391s|195M/s|
|52|27| +0| +0|A3| 607M|   2.527s|240M/s|
|53|28| -2| -2|D8|3.26G|   9.914s|329M/s|
|54|28| -2| -2|C7|8.52G|  25.594s|333M/s|
|55|29| +0| +0|G6|22.1G|  71.858s|308M/s|
|56|29| +2| +2|H5|2.36G|  10.000s|236M/s|
|57|30|-10|-10|A6|12.2G|  45.150s|270M/s|
|58|30| +4| +4|G1|2.80G|  11.297s|248M/s|
|59|34|+64|+64|G8|   97|   0.037s|0M/s|

FFO 60-73

|No.|empties|result|answer|move|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|:--:|---:|
|60|24|+20|+20|C2|65.5M|   0.233s|280M/s|
|61|25|-14|-14|G1| 171M|   0.621s|276M/s|
|62|27|+28|+28|E8|4.23G|  15.926s|266M/s|
|63|27| -2| -2|F2|1.88G|   6.533s|289M/s|
|64|27|+20|+20|B4|5.79G|  24.549s|236M/s|
|65|28|+10|+10|G1|25.3G| 100.981s|250M/s|
|66|28|+30|+30|H3|14.4G|  37.929s|381M/s|
|67|28|+22|+22|H3|18.0G|  56.097s|322M/s|
|68|30|+28|+28|E8| 135G| 265.413s|511M/s|
|69|30| +0| +0|H3|16.3G|  58.738s|278M/s|
|70|30|-24|-24|E3|7.10G|  34.440s|206M/s|
|71|31|+20|+20|D2|16.9G|  75.218s|224M/s|
|72|31|+24|+24|E1| 227G| 656.243s|346M/s|
|73|31| -4| -4|G4|20.9G|  87.895s|238M/s|
