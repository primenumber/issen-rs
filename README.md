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

- Date: 2023/12/01
- Hardware: AMD Ryzen 9 7950X3D, DDR5-4800 64GB
- Environment: Linux 5.15.0-89-generic, Ubuntu 22.04.3, rustc 1.75.0-nightly

FFO 40-59

|No.|empties|result|answer|move|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|:--:|---:|
|40|20|+38|+38|A2|89.8M|   0.056s|1576M/s|
|41|22| +0| +0|H4| 106M|   0.079s|1332M/s|
|42|22| +6| +6|G2| 280M|   0.165s|1691M/s|
|43|23|-12|-12|C7| 190M|   0.146s|1294M/s|
|44|23|-14|-14|D2| 113M|   0.085s|1323M/s|
|45|24| +6| +6|B2|1.59G|   1.006s|1580M/s|
|46|24| -8| -8|B3| 502M|   0.351s|1427M/s|
|47|25| +4| +4|G2| 167M|   0.124s|1338M/s|
|48|25|+28|+28|F6| 954M|   0.721s|1321M/s|
|49|26|+16|+16|E1|3.00G|   1.905s|1576M/s|
|50|26|+10|+10|D8|3.49G|   2.752s|1270M/s|
|51|27| +6| +6|E2|1.52G|   1.255s|1210M/s|
|52|27| +0| +0|A3|1.40G|   1.145s|1229M/s|
|53|28| -2| -2|D8|5.41G|   4.412s|1226M/s|
|54|28| -2| -2|C7|15.5G|  10.756s|1441M/s|
|55|29| +0| +0|G6|30.3G|  29.226s|1039M/s|
|56|29| +2| +2|H5|5.31G|   5.211s|1019M/s|
|57|30|-10|-10|A6|24.2G|  22.297s|1089M/s|
|58|30| +4| +4|G1|6.53G|   6.084s|1074M/s|
|59|34|+64|+64|G8|2.00k|   0.041s|0M/s|

[Total] elapsed: 87828878us, node count: 100876738519, NPS: 1148560027nodes/sec

FFO 60-79

|No.|empties|result|answer|move|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|:--:|---:|
|60|24|+20|+20|C2| 211M|   0.159s|1324M/s|
|61|25|-14|-14|G1| 335M|   0.303s|1102M/s|
|62|27|+28|+28|E8|8.08G|   6.938s|1164M/s|
|63|27| -2| -2|F2|2.89G|   2.381s|1213M/s|
|64|27|+20|+20|B4|11.6G|   9.471s|1232M/s|
|65|28|+10|+10|G1|31.1G|  21.178s|1470M/s|
|66|28|+30|+30|H3|22.5G|  16.607s|1356M/s|
|67|28|+22|+22|H3|28.9G|  20.325s|1425M/s|
|68|30|+28|+28|E8| 153G| 115.037s|1335M/s|
|69|30| +0| +0|H3|18.2G|  15.637s|1164M/s|
|70|30|-24|-24|E3|16.1G|  13.649s|1182M/s|
|71|31|+20|+20|D2|27.6G|  25.330s|1091M/s|
|72|31|+24|+24|E1| 254G| 284.996s|892M/s|
|73|31| -4| -4|G4|34.2G|  38.391s|890M/s|
|74|31|-30|-30|F1| 846G| 762.676s|1109M/s|
|75|32|+14|+14|D2| 334G| 241.491s|1384M/s|
|76|32|+32|+32|A3|2.39T|2046.156s|1171M/s|
|77|34|+34|+34|B7|2.08T|1720.431s|1210M/s|
|78|34| +8| +8|F1|1.04T|1231.807s|845M/s|
|79|36|+64|+64|D7|28.2G|  33.405s|846M/s|

[Total] elapsed: 6606379988us, node count: 7340282057714, NPS: 1111089896nodes/sec
