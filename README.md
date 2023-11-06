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

- Date: 2023/11/06
- Hardware: AMD Ryzen 9 5950X, DDR4-3200 128GB
- Environment: Linux 5.15.90.1-microsoft-standard-WSL2, Ubuntu 22.04.2 on WSL2, rustc 1.75.0-nightly

FFO 40-59

|No.|empties|result|answer|move|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|:--:|---:|
|40|20|+38|+38|A2|26.4M|   0.107s|245M/s|
|41|22| +0| +0|H4|30.6M|   0.142s|214M/s|
|42|22| +6| +6|G2|68.3M|   0.268s|253M/s|
|43|23|-12|-12|C7|60.7M|   0.275s|220M/s|
|44|23|-14|-14|D2|33.9M|   0.130s|258M/s|
|45|24| +6| +6|B2|1.04G|   2.509s|415M/s|
|46|24| -8| -8|B3| 151M|   0.678s|222M/s|
|47|25| +4| +4|G2|36.0M|   0.157s|228M/s|
|48|25|+28|+28|F6| 451M|   1.273s|354M/s|
|49|26|+16|+16|E1|1.90G|   4.383s|434M/s|
|50|26|+10|+10|D8|2.51G|   5.464s|460M/s|
|51|27| +6| +6|A3| 661M|   3.237s|204M/s|
|52|27| +0| +0|A3| 563M|   1.846s|305M/s|
|53|28| -2| -2|D8|3.06G|   8.751s|350M/s|
|54|28| -2| -2|C7|8.67G|  23.092s|375M/s|
|55|29| +0| +0|G6|22.3G|  72.688s|307M/s|
|56|29| +2| +2|H5|2.42G|  10.441s|231M/s|
|57|30|-10|-10|A6|12.1G|  42.238s|288M/s|
|58|30| +4| +4|G1|2.89G|  11.296s|256M/s|
|59|34|+64|+64|G8|   97|   0.029s|0M/s|

FFO 60-73

|No.|empties|result|answer|move|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|:--:|---:|
|60|24|+20|+20|C2|62.7M|   0.214s|291M/s|
|61|25|-14|-14|G1| 169M|   0.540s|312M/s|
|62|27|+28|+28|E8|4.22G|  13.912s|303M/s|
|63|27| -2| -2|F2|1.81G|   5.464s|331M/s|
|64|27|+20|+20|B4|5.93G|  19.493s|304M/s|
|65|28|+10|+10|G1|28.6G|  84.079s|340M/s|
|66|28|+30|+30|H3|14.4G|  34.743s|417M/s|
|67|28|+22|+22|H3|19.1G|  46.115s|416M/s|
|68|30|+28|+28|E8| 133G| 234.547s|571M/s|
|69|30| +0| +0|H3|15.9G|  51.676s|309M/s|
|70|30|-24|-24|E3|7.31G|  26.322s|277M/s|
|71|31|+20|+20|D2|15.5G|  67.077s|231M/s|
|72|31|+24|+24|E1| 231G| 673.563s|344M/s|
|73|31| -4| -4|G4|20.9G|  87.712s|238M/s|
