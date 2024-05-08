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

- Date: 2024/03/29
- Hardware: AMD Ryzen 9 7950X3D, DDR5-4800 64GB
- Environment: Linux 6.5.0-26-generic, Ubuntu 22.04.4, rustc 1.78.0-nightly

FFO 40-59

|No.|empties|result|answer|move|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|:--:|---:|
|40|20|+38|+38|A2|76.2M|   0.062s|1209M/s|
|41|22| +0| +0|H4|87.2M|   0.080s|1076M/s|
|42|22| +6| +6|G2| 270M|   0.178s|1511M/s|
|43|23|-12|-12|C7| 164M|   0.146s|1122M/s|
|44|23|-14|-14|D2|84.3M|   0.081s|1029M/s|
|45|24| +6| +6|B2|1.51G|   0.978s|1547M/s|
|46|24| -8| -8|B3| 443M|   0.329s|1343M/s|
|47|25| +4| +4|G2| 133M|   0.122s|1084M/s|
|48|25|+28|+28|F6| 874M|   0.676s|1291M/s|
|49|26|+16|+16|E1|3.14G|   2.047s|1536M/s|
|50|26|+10|+10|D8|3.24G|   2.543s|1276M/s|
|51|27| +6| +6|E2|1.41G|   1.188s|1191M/s|
|52|27| +0| +0|A3|1.42G|   1.175s|1208M/s|
|53|28| -2| -2|D8|4.91G|   4.038s|1216M/s|
|54|28| -2| -2|C7|13.6G|   9.923s|1375M/s|
|55|29| +0| +0|G6|27.3G|  25.529s|1070M/s|
|56|29| +2| +2|H5|4.26G|   4.629s|921M/s|
|57|30|-10|-10|A6|19.6G|  18.350s|1071M/s|
|58|30| +4| +4|G1|4.48G|   4.760s|942M/s|
|59|34|+64|+64|G8|1.26k|   0.029s|0M/s|

[Total] elapsed: 76872921us, node count: 87193095283, NPS: 1134249800nodes/sec

FFO 60-79

|No.|empties|result|answer|move|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|:--:|---:|
|60|24|+20|+20|C2| 213M|   0.175s|1213M/s|
|61|25|-14|-14|G1| 344M|   0.325s|1058M/s|
|62|27|+28|+28|E8|7.61G|   6.587s|1155M/s|
|63|27| -2| -2|F2|2.89G|   2.383s|1215M/s|
|64|27|+20|+20|B4|10.4G|   8.675s|1204M/s|
|65|28|+10|+10|G1|26.6G|  18.953s|1405M/s|
|66|28|+30|+30|H3|19.7G|  15.215s|1299M/s|
|67|28|+22|+22|H3|25.8G|  18.823s|1375M/s|
|68|30|+28|+28|E8| 116G|  90.898s|1287M/s|
|69|30| +0| +0|H3|14.5G|  13.385s|1086M/s|
|70|30|-24|-24|E3|13.0G|  11.413s|1146M/s|
|71|31|+20|+20|D2|21.3G|  21.350s|999M/s|
|72|31|+24|+24|E1| 178G| 197.643s|902M/s|
|73|31| -4| -4|G4|25.8G|  31.323s|824M/s|
|74|31|-30|-30|F1| 599G| 580.678s|1031M/s|
|75|32|+14|+14|D2| 234G| 180.377s|1300M/s|
|76|32|+32|+32|A3|1.56T|1441.196s|1087M/s|
|77|34|+34|+34|B7|1.23T|1122.816s|1100M/s|
|78|34| +8| +8|F1| 571G| 831.075s|687M/s|
|79|36|+64|+64|D7|14.6G|  12.269s|1191M/s|

[Total] elapsed: 4605575684us, node count: 4685891940872, NPS: 1017438918nodes/sec
