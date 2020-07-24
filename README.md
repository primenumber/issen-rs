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
|40|20|+38|+38|54.0M|   0.3s|177M/s|
|41|22| +0| +0| 163M|   0.6s|255M/s|
|42|22| +6| +6|98.4M|   0.4s|230M/s|
|43|23|-12|-12| 467M|   1.8s|254M/s|
|44|23|-14|-14| 510M|   1.5s|326M/s|
|45|24| +6| +6|4.79G|  13.1s|364M/s|
|46|24| -8| -8| 669M|   2.0s|324M/s|
|47|25| +4| +4| 145M|   0.8s|166M/s|
|48|25|+28|+28|2.13G|   5.4s|391M/s|
|49|26|+16|+16|3.63G|  11.3s|320M/s|
|50|26|+10|+10|9.12G|  30.2s|301M/s|
|51|27| +6| +6|4.65G|  14.9s|310M/s|
|52|27| +0| +0|3.96G|  14.4s|274M/s|
|53|28| -2| -2|39.2G| 116.1s|338M/s|
|54|28| -2| -2|30.5G|  89.8s|340M/s|
|55|29| +0| +0| 164G| 467.8s|352M/s|
|56|29| +2| +2|5.98G|  31.8s|187M/s|
|57|30|-10|-10|30.8G| 144.8s|212M/s|
|58|30| +4| +4|12.5G|  66.0s|190M/s|
|59|34|+64|+64|8.13G| 119.7s|67M/s|
