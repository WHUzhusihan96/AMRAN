# AMRAN
A deep domain adaptation method for remote sensing cross-scene classification.

## Requirement
* python 3
* pytorch 1.0

## Usage
1. The `root_path` is the directory of you dataset. And the dataset please placed as: `./dataset/domain/classes/samples`
2. You can change the `source_dir` and `test_dir` in `AMRAN.py` to set different transfer tasks.
3. Run `python AMRAN.py`

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
