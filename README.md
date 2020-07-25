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

Date: 2020/07/25
Hardware: AMD Ryzen 9 3900X, DDR4-3200 64GB
Environment: Linux 5.4.0-42, Ubuntu 20.04, rustc 1.46.0-nightly

FFO 40-59

|No.|empties|result|answer|nodes|time|NPS|
|----|----|----|----|----|----|----|
|40|20|+38|+38|45.0M|   0.264s|169M/s|
|41|22| +0| +0| 120M|   0.445s|270M/s|
|42|22| +6| +6|82.7M|   0.368s|224M/s|
|43|23|-12|-12| 340M|   1.152s|295M/s|
|44|23|-14|-14| 384M|   1.008s|380M/s|
|45|24| +6| +6|3.33G|   7.299s|456M/s|
|46|24| -8| -8| 525M|   1.454s|361M/s|
|47|25| +4| +4| 117M|   0.651s|179M/s|
|48|25|+28|+28|1.77G|   3.909s|453M/s|
|49|26|+16|+16|2.84G|   7.693s|369M/s|
|50|26|+10|+10|6.55G|  16.656s|393M/s|
|51|27| +6| +6|3.81G|  11.324s|337M/s|
|52|27| +0| +0|3.24G|  11.490s|282M/s|
|53|28| -2| -2|33.3G|  91.564s|363M/s|
|54|28| -2| -2|25.5G|  71.371s|357M/s|
|55|29| +0| +0| 153G| 398.665s|384M/s|
|56|29| +2| +2|4.95G|  25.483s|194M/s|
|57|30|-10|-10|27.0G| 123.774s|218M/s|
|58|30| +4| +4|10.2G|  50.656s|203M/s|
|59|34|+64|+64|7.01G| 100.979s|69M/s|

FFO 60-69

|No.|empties|result|answer|nodes|time|NPS|
|----|----|----|----|----|----|----|
|60|24|+20|+20| 144M|   0.621s|232M/s|
|61|25|-14|-14| 666M|   2.209s|301M/s|
|62|27|+28|+28|14.8G|  44.450s|333M/s|
|63|27| -2| -2|8.25G|  23.018s|358M/s|
|64|27|+20|+20|83.3G| 184.028s|453M/s|
|65|28|+10|+10| 125G| 279.933s|447M/s|
|66|28|+30|+30|45.8G| 121.801s|376M/s|
|67|28|+22|+22|98.5G| 233.340s|422M/s|
|68|30|+28|+28| 265G| 671.746s|395M/s|
|69|30| +0| +0| 118G| 338.676s|349M/s|
