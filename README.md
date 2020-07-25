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

Date: 2020/07/26
Hardware: AMD Ryzen 9 3900X, DDR4-3200 64GB
Environment: Linux 5.4.0-42, Ubuntu 20.04, rustc 1.46.0-nightly

FFO 40-59

|No.|empties|result|answer|nodes|time|NPS|
|----|----|----|----|----|----|----|
|40|20|+38|+38|34.6M|   0.241s|143M/s|
|41|22| +0| +0|73.0M|   0.364s|200M/s|
|42|22| +6| +6|51.3M|   0.269s|190M/s|
|43|23|-12|-12| 243M|   0.948s|256M/s|
|44|23|-14|-14| 208M|   0.715s|290M/s|
|45|24| +6| +6|1.84G|   5.325s|346M/s|
|46|24| -8| -8| 319M|   1.088s|293M/s|
|47|25| +4| +4|58.7M|   0.325s|180M/s|
|48|25|+28|+28|1.08G|   2.910s|374M/s|
|49|26|+16|+16|1.69G|   5.795s|292M/s|
|50|26|+10|+10|5.87G|  13.650s|430M/s|
|51|27| +6| +6|1.68G|   8.127s|207M/s|
|52|27| +0| +0|1.37G|   6.470s|213M/s|
|53|28| -2| -2|17.0G|  66.251s|257M/s|
|54|28| -2| -2|18.8G|  73.093s|258M/s|
|55|29| +0| +0|80.8G| 293.191s|275M/s|
|56|29| +2| +2|4.22G|  25.389s|166M/s|
|57|30|-10|-10|18.3G| 103.126s|177M/s|
|58|30| +4| +4|8.25G|  52.866s|156M/s|
|59|34|+64|+64|9.71G| 150.761s|64M/s|

FFO 60-69

|No.|empties|result|answer|nodes|time|NPS|
|----|----|----|----|----|----|----|
|60|24|+20|+20| 100M|   0.452s|221M/s|
|61|25|-14|-14| 478M|   1.887s|253M/s|
|62|27|+28|+28|6.73G|  36.795s|183M/s|
|63|27| -2| -2|4.80G|  17.408s|276M/s|
|64|27|+20|+20|23.2G|  93.194s|249M/s|
|65|28|+10|+10|35.2G| 154.564s|227M/s|
|66|28|+30|+30|34.1G|  99.507s|343M/s|
|67|28|+22|+22|54.8G| 145.148s|377M/s|
|68|30|+28|+28| 319G| 688.480s|464M/s|
|69|30| +0| +0|66.6G| 238.937s|278M/s|
