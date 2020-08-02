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

Date: 2020/08/03
Hardware: AMD Ryzen 9 3900X, DDR4-3200 64GB
Environment: Linux 5.4.0-42, Ubuntu 20.04.1, rustc 1.46.0-nightly

FFO 40-59

|No.|empties|result|answer|nodes|time|NPS|
|----|----|----|----|----|----|----|
|40|20|+38|+38|32.5M|   0.245s|132M/s|
|41|22| +0| +0|58.9M|   0.341s|172M/s|
|42|22| +6| +6|48.8M|   0.279s|174M/s|
|43|23|-12|-12| 204M|   0.950s|215M/s|
|44|23|-14|-14| 168M|   0.660s|254M/s|
|45|24| +6| +6|1.65G|   5.045s|328M/s|
|46|24| -8| -8| 249M|   1.028s|242M/s|
|47|25| +4| +4|48.8M|   0.297s|163M/s|
|48|25|+28|+28| 849M|   2.806s|302M/s|
|49|26|+16|+16|1.56G|   5.506s|284M/s|
|50|26|+10|+10|5.91G|  14.046s|421M/s|
|51|27| +6| +6|1.33G|   7.676s|173M/s|
|52|27| +0| +0|1.02G|   5.919s|173M/s|
|53|28| -2| -2|13.8G|  62.406s|221M/s|
|54|28| -2| -2|16.3G|  68.561s|238M/s|
|55|29| +0| +0|66.5G| 284.129s|234M/s|
|56|29| +2| +2|3.41G|  24.626s|138M/s|
|57|30|-10|-10|14.6G| 102.607s|142M/s|
|58|30| +4| +4|6.88G|  51.257s|134M/s|
|59|34|+64|+64|8.78G| 146.726s|59M/s|

FFO 60-69

|No.|empties|result|answer|nodes|time|NPS|
|----|----|----|----|----|----|----|
|60|24|+20|+20|86.4M|   0.435s|198M/s|
|61|25|-14|-14| 380M|   1.791s|212M/s|
|62|27|+28|+28|4.97G|  35.935s|138M/s|
|63|27| -2| -2|3.84G|  16.921s|227M/s|
|64|27|+20|+20|18.1G|  92.550s|195M/s|
|65|28|+10|+10|33.1G| 155.093s|213M/s|
|66|28|+30|+30|32.7G|  99.416s|329M/s|
|67|28|+22|+22|48.8G| 144.519s|338M/s|
|68|30|+28|+28| 317G| 706.330s|450M/s|
|69|30| +0| +0|52.6G| 222.776s|236M/s|
