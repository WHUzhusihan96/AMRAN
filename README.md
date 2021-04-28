# AMRAN
A deep domain adaptation method for remote sensing cross-scene classification.

## Requirement
* python 3
* pytorch 1.0

## Dataset
Add the dataset selected in paper to this repo. [[BaiduYun](https://pan.baidu.com/s/1WOuXxOe1mav9cKdESsCXDw)] 密码:szu0

You also can use your own data.

## Usage
1. The `root_path` is the directory of you dataset. And the dataset please placed as: `./dataset/domain/classes/samples`
2. You can change the `source_dir` and `test_dir` in `AMRAN.py` to set different transfer tasks.
3. Run `python AMRAN.py`

## Results on RSdataset
|     Method     |   U-W  |   U-A  |   U-R  |   W-U  |   W-A  |   W-R  |   A-U  |   A-W  |   A-R  |   R-U  |   R-W  |   R-A  |   AVG  |
|:--------------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|     AlexNet    | 63.80  | 58.56  | 51.67  | 73.31  | 86.49  | 68.82  | 64.84  | 96.20  | 73.22  | 78.09  | 86.71  | 81.65  | 73.61  |
| AMRAN(AlexNet) | 81.46  | 75.63  | 66.39  | 80.66  | 87.39  | 74.71  | 70.74  | 96.52  | 76.63  | 82.86  | 88.29  | 85.14  | 80.53  |
|    ResNet50    | 71.20  | 66.44  | 56.50  | 75.14  | 93.11  | 71.25  | 74.86  | 98.42  | 76.88  | 80.71  | 86.71  | 85.18  | 78.03  |
|    AMRANavg    | 89.56  | 84.80  | 69.02  | 85.20  | 94.82  | 79.50  | 88.16  | 99.68  | 81.00  | 87.56  | 94.59  | 94.48  | 87.36  |
|    AMRANmax    | 89.87  | 86.89  | 69.75  | 87.14  | 95.72  | 80.29  | 90.00  | 99.68  | 81.75  | 89.86  | 94.94  | 94.59  | 88.37  |

> AMRANavg and AMRANmax are based on the ResNet50.
> 
## Reference

```
S. Zhu, B. Du, L. Zhang and X. Li, "Attention-Based Multiscale Residual Adaptation Network for Cross-Scene Classification," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3056624.
```

or in bibtex style:

```
@ARTICLE{9377566,
  author={S. {Zhu} and B. {Du} and L. {Zhang} and X. {Li}},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Attention-Based Multiscale Residual Adaptation Network for Cross-Scene Classification}, 
  year={2021},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2021.3056624}}
```
