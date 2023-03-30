### Dataset

Download datasets and place them in 'datasets' folder in the following structure:
- [MF dataset](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) or [RTFNet preprocessed version](http://gofile.me/4jm56/CfukComo1)
- [PST900 dataset](https://github.com/ShreyasSkandanS/pst900_thermal_rgb)
- [KP dataset](https://github.com/SoonminHwang/rgbt-ped-detection), [Segmentation label](https://github.com/yeong5366/MS-UDA) or [Pre-organized KP dataset](https://github.com/yeong5366/MS-UDA) or [Pre-organized KP dataset](https://kaistackr-my.sharepoint.com/:u:/g/personal/shinwc159_kaist_ac_kr/EUfmm7hkeaVNuyyYsREttFIBGZ3u_tCmaZ5S5EYghwkKnQ?e=Gyc86F)

Since the original KP dataset has a large volume (>35GB) and requesting labels takes time, we recommend to use our pre-organized KP dataset (includes labels as well).

```shell
<datasets>
|-- <MFdataset>
    |-- <images>
    |-- <labels>
    |-- train.txt
    |-- val.txt
    |-- test.txt
    ...
|-- <PSTdataset>
    |-- <train>
        |-- rgb
        |-- thermal
        |-- labels
        ...
    |-- <test>
        |-- rgb
        |-- thermal
        |-- labels
        ...
|-- <KPataset>
    |-- <images>
        |-- set00
        |-- set01
        ...
    |-- <labels>
    |-- train.txt
    |-- val.txt
    |-- test.txt
    ...
```
