# MOS: Modeling Object-Scene Associations in Generalized Category Discovery (CVPR 2025)

<p align="center">
    <a href="https://arxiv.org/abs/2503.12035"><img src="https://img.shields.io/badge/arXiv-2503.12035-b31b1b"></a>
    <a href="https://github.com/CVMI-Lab/SlotCon/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
</p>

Welcome to the official repository for the  [MOS: Modeling Object-Scene Associations in Generalized Category Discovery](https://arxiv.org/abs/2503.12035) project!



**Important Note:**  
The code is currently under organization, and we commit to open-sourcing it before 2025.5.1. If you require immediate access, please contact us directly at: pengzhengyuan@sjtu.edu.cn.

## Running

### Dependencies

```
pip install -r requirements.txt
```

### Scripts

**Train the model**:

```
bash scripts/run_${DATASET_NAME}.sh
```

### Datasets

We use fine-grained benchmarks in this paper, including:

* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## Citing this work

If you find this repo useful for your research, please consider citing our paper:

```
@article{peng2025mos,
  title={MOS: Modeling Object-Scene Associations in Generalized Category Discovery},
  author={Peng, Zhengyuan and Ma, Jinpeng and Sun, Zhimin and Yi, Ran and Song, Haichuan and Tan, Xin and Ma, Lizhuang},
  journal={arXiv preprint arXiv:2503.12035},
  year={2025}
}
```

## Acknowledgements

The codebase is largely built on this repo: [SimGCD](https://github.com/CVMI-Lab/SimGCD?tab=readme-ov-file).

## Contact

For inquiries or further information, contact: pengzhengyuan@sjtu.edu.cn

Happy coding!
