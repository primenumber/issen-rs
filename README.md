# issen-rs

Reversi solver on Rust

## License

GPLv3

## prerequirements

AVX2 and BMI2 enabled x86\_64 CPU (Intel: Haswell or later, AMD: Excavator or later)

## Setup

prepare evaluation table

```Shell
$ wget https://poyo.me/reversi/table210925.tar.gz
$ tar xf table210925.tar.gz
```

## Run

compile and run

```Shell
$ cargo run --release
```

## Benchmark result

Date: 2020/08/08
Hardware: AMD Ryzen 9 3900X, DDR4-3200 64GB
Environment: Linux 5.4.0-42, Ubuntu 20.04.1, rustc 1.46.0-nightly

FFO 40-59

|No.|empties|result|answer|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|---:|
|40|20|+38|+38|30.7M|   0.209s|146M/s|
|41|22| +0| +0|59.5M|   0.319s|185M/s|
|42|22| +6| +6|37.6M|   0.224s|167M/s|
|43|23|-12|-12| 246M|   0.874s|281M/s|
|44|23|-14|-14| 169M|   0.744s|227M/s|
|45|24| +6| +6|1.66G|   5.111s|324M/s|
|46|24| -8| -8| 256M|   1.006s|254M/s|
|47|25| +4| +4|50.7M|   0.308s|164M/s|
|48|25|+28|+28| 916M|   2.919s|314M/s|
|49|26|+16|+16|1.59G|   5.361s|297M/s|
|50|26|+10|+10|5.89G|  13.501s|436M/s|
|51|27| +6| +6|1.41G|   7.246s|195M/s|
|52|27| +0| +0|1.03G|   5.690s|182M/s|
|53|28| -2| -2|14.1G|  61.533s|229M/s|
|54|28| -2| -2|16.3G|  64.340s|254M/s|
|55|29| +0| +0|59.0G| 236.544s|249M/s|
|56|29| +0| +2|3.27G|  23.417s|139M/s|
|57|30|-10|-10|14.5G|  96.030s|151M/s|
|58|30| +4| +4|6.91G|  46.134s|149M/s|
|59|34|+64|+64|8.80G| 127.889s|68M/s|

FFO 60-69

|No.|empties|result|answer|nodes|time|NPS|
|---:|---:|---:|---:|---:|---:|---:|
|60|24|+20|+20|87.8M|   0.425s|206M/s|
|61|25|-14|-14| 380M|   1.601s|237M/s|
|62|27|+28|+28|5.14G|  32.814s|156M/s|
|63|27| -2| -2|4.08G|  18.038s|226M/s|
|64|27|+20|+20|19.1G|  89.721s|213M/s|
|65|28|+10|+10|34.0G| 158.913s|214M/s|
|66|28|+30|+30|32.0G|  98.182s|326M/s|
|67|28|+22|+22|51.0G| 143.298s|356M/s|
|68|30|+28|+28| 316G| 682.832s|463M/s|
|69|30| +0| +0|53.4G| 217.642s|245M/s|
