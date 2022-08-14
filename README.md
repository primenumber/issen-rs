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

- Date: 2022/08/15
- Hardware: AMD Ryzen 9 5950X, DDR4-3200 128GB
- Environment: Linux 5.10.102.1-microsoft-standard-WSL2, Ubuntu 20.04.4 on WSL2, rustc 1.64.0-nightly

FFO 40-59

|No.|empties|result|answer|move|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|:--:|---:|
|40|20|+38|+38|A2|26.2M|   0.111s|234M/s|
|41|22| +0| +0|H4|29.8M|   0.170s|174M/s|
|42|22| +6| +6|G2|68.9M|   0.318s|216M/s|
|43|23|-12|-12|C7|90.4M|   0.310s|290M/s|
|44|23|-14|-14|D2|33.3M|   0.162s|204M/s|
|45|24| +6| +6|B2| 695M|   1.473s|471M/s|
|46|24| -8| -8|B3| 175M|   0.780s|225M/s|
|47|25| +4| +4|G2|35.4M|   0.159s|221M/s|
|48|25|+28|+28|F6| 592M|   1.559s|379M/s|
|49|26|+16|+16|E1|2.19G|   6.148s|356M/s|
|50|26|+10|+10|D8|2.56G|   5.398s|475M/s|
|51|27| +6| +6|A3| 453M|   2.202s|205M/s|
|52|27| +0| +0|A3| 572M|   2.195s|260M/s|
|53|28| -2| -2|D8|3.34G|   9.984s|335M/s|
|54|28| -2| -2|C7|9.97G|  26.314s|378M/s|
|55|29| +0| +0|B7|30.4G|  77.545s|392M/s|
|56|29| +2| +2|H5|2.48G|  10.067s|246M/s|
|57|30|-10|-10|A6|28.5G|  99.465s|287M/s|
|58|30| +4| +4|G1|4.13G|  14.933s|276M/s|
|59|34|+64|+64|G8|   97|   0.038s|0M/s|

FFO 60-73

|No.|empties|result|answer|move|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|:--:|---:|
|60|24|+20|+20|C2|63.6M|   0.221s|286M/s|
|61|25|-14|-14|G1| 168M|   0.584s|288M/s|
|62|27|+28|+28|E8|4.73G|  15.139s|312M/s|
|63|27| -2| -2|F2|2.00G|   6.239s|321M/s|
|64|27|+20|+20|B4|7.46G|  24.172s|308M/s|
|65|28|+10|+10|G1|40.1G| 125.817s|319M/s|
|66|28|+30|+30|H3|16.7G|  40.461s|414M/s|
|67|28|+22|+22|H3|32.8G|  82.992s|395M/s|
|68|30|+28|+28|E8| 150G| 261.200s|576M/s|
|69|30| +0| +0|H3|32.4G| 118.993s|272M/s|
|70|30|-24|-24|E3|8.43G|  31.474s|267M/s|
|71|31|+20|+20|D2|15.6G|  56.073s|279M/s|
|72|31|+24|+24|E1| 154G| 426.226s|363M/s|
|73|31| -4| -4|G4|21.9G|  82.854s|265M/s|
