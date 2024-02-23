# issen-rs

Reversi solver on Rust

## License

GPLv3

## prerequirements

~~AVX2 and BMI2 enabled x86\_64 CPU (Intel: Haswell or later, AMD: Excavator or later)~~
issen-rs is now portable! We have confirmed that it works on x86\_64 (AMD64) and AArch64.

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

- Date: 2024/02/21
- Hardware: AMD Ryzen 9 7950X3D, DDR5-4800 64GB
- Environment: Linux 6.5.0-18-generic, Ubuntu 22.04.4, rustc 1.78.0-nightly

FFO 40-59

|No.|empties|result|answer|move|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|:--:|---:|
|40|20|+38|+38|A2|90.4M|   0.054s|1644M/s|
|41|22| +0| +0|H4| 117M|   0.092s|1267M/s|
|42|22| +6| +6|G2| 287M|   0.177s|1617M/s|
|43|23|-12|-12|C7| 159M|   0.121s|1303M/s|
|44|23|-14|-14|D2| 111M|   0.083s|1322M/s|
|45|24| +6| +6|B2|1.57G|   1.010s|1561M/s|
|46|24| -8| -8|B3| 494M|   0.349s|1412M/s|
|47|25| +4| +4|G2| 167M|   0.123s|1350M/s|
|48|25|+28|+28|F6| 901M|   0.680s|1323M/s|
|49|26|+16|+16|E1|3.10G|   1.988s|1559M/s|
|50|26|+10|+10|D8|3.56G|   2.773s|1285M/s|
|51|27| +6| +6|E2|1.47G|   1.240s|1192M/s|
|52|27| +0| +0|A3|1.31G|   1.080s|1215M/s|
|53|28| -2| -2|D8|5.77G|   4.829s|1196M/s|
|54|28| -2| -2|C7|15.6G|  11.420s|1368M/s|
|55|29| +0| +0|G6|29.2G|  28.391s|1030M/s|
|56|29| +2| +2|H5|4.56G|   4.727s|965M/s|
|57|30|-10|-10|A6|19.8G|  18.593s|1066M/s|
|58|30| +4| +4|G1|4.83G|   4.961s|973M/s|
|59|34|+64|+64|G8|1.66k|   0.034s|0M/s|

[Total] elapsed: 82738454us, node count: 93254297417, NPS: 1127097412nodes/sec

FFO 60-79

|No.|empties|result|answer|move|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|:--:|---:|
|60|24|+20|+20|C2| 216M|   0.164s|1313M/s|
|61|25|-14|-14|G1| 339M|   0.318s|1062M/s|
|62|27|+28|+28|E8|8.51G|   7.369s|1154M/s|
|63|27| -2| -2|F2|2.88G|   2.393s|1203M/s|
|64|27|+20|+20|B4|11.2G|   9.350s|1207M/s|
|65|28|+10|+10|G1|29.2G|  20.527s|1422M/s|
|66|28|+30|+30|H3|21.1G|  16.079s|1314M/s|
|67|28|+22|+22|H3|28.2G|  20.363s|1386M/s|
|68|30|+28|+28|E8| 139G| 107.049s|1301M/s|
|69|30| +0| +0|H3|15.7G|  14.081s|1121M/s|
|70|30|-24|-24|E3|14.4G|  13.097s|1106M/s|
|71|31|+20|+20|D2|20.1G|  19.281s|1044M/s|
|72|31|+24|+24|E1| 258G| 298.006s|867M/s|
|73|31| -4| -4|G4|29.4G|  34.672s|849M/s|
|74|31|-30|-30|F1| 611G| 563.534s|1084M/s|
|75|32|+14|+14|D2| 299G| 225.346s|1327M/s|
|76|32|+32|+32|A3|2.12T|1869.331s|1134M/s|
|77|34|+34|+34|B7|1.13T|1015.754s|1121M/s|
|78|34| +8| +8|F1| 673G| 844.518s|797M/s|
|79|36|+64|+64|D7|56.8G|  43.134s|1319M/s|

[Total] elapsed: 5124378900us, node count: 5480838886501, NPS: 1069561598nodes/sec
