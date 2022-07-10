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

- Date: 2022/07/10
- Hardware: AMD Ryzen 9 5950X, DDR4-3200 128GB
- Environment: Linux 5.10.102.1-microsoft-standard-WSL2, Ubuntu 20.04.4 on WSL2, rustc 1.64.0-nightly

FFO 40-59

|No.|empties|result|answer|move|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|:--:|---:|
|40|20|+38|+38|A2|26.6M|   0.112s|235M/s|
|41|22| +0| +0|H4|29.8M|   0.168s|176M/s|
|42|22| +6| +6|G2|67.8M|   0.316s|213M/s|
|43|23|-12|-12|C7| 101M|   0.309s|327M/s|
|44|23|-14|-14|D2|33.4M|   0.160s|207M/s|
|45|24| +6| +6|B2| 703M|   1.457s|482M/s|
|46|24| -8| -8|B3| 176M|   0.768s|230M/s|
|47|25| +4| +4|G2|34.9M|   0.160s|216M/s|
|48|25|+28|+28|F6| 675M|   1.741s|387M/s|
|49|26|+16|+16|E1|2.19G|   6.076s|361M/s|
|50|26|+10|+10|D8|2.56G|   5.386s|475M/s|
|51|27| +6| +6|A3| 442M|   2.050s|215M/s|
|52|27| +0| +0|A3| 569M|   2.180s|261M/s|
|53|28| -2| -2|D8|3.33G|   9.921s|336M/s|
|54|28| -2| -2|C7|9.99G|  26.639s|375M/s|
|55|29| +0| +0|B7|30.6G|  76.960s|398M/s|
|56|29| +2| +2|H5|2.45G|   9.965s|245M/s|
|57|30|-10|-10|A6|28.3G|  96.916s|292M/s|
|58|30| +4| +4|G1|4.14G|  14.676s|282M/s|
|59|34|+64|+64|G8|   97|   0.037s|0M/s|

FFO 60-73

|No.|empties|result|answer|move|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|:--:|---:|
|60|24|+20|+20|C2|63.6M|   0.216s|293M/s|
|61|25|-14|-14|G1| 168M|   0.571s|295M/s|
|62|27|+28|+28|E8|4.84G|  15.162s|319M/s|
|63|27| -2| -2|F2|1.94G|   6.139s|317M/s|
|64|27|+20|+20|B4|7.48G|  23.982s|311M/s|
|65|28|+10|+10|G1|40.0G| 126.521s|316M/s|
|66|28|+30|+30|H3|19.1G|  43.469s|440M/s|
|67|28|+22|+22|H3|32.7G|  81.106s|403M/s|
|68|30|+28|+28|E8| 180G| 310.511s|581M/s|
|69|30| +0| +0|H3|32.4G| 124.821s|259M/s|
|70|30|-24|-24|E3|8.54G|  29.472s|289M/s|
|71|31|+20|+20|D2|15.6G|  55.674s|281M/s|
|72|31|+24|+24|E1| 157G| 423.859s|370M/s|
|73|31| -4| -4|G4|22.3G|  81.668s|273M/s|
