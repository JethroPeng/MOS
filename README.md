# MOS: Modeling Object-Scene Associations in Generalized Category Discovery (CVPR 2025)

<p align="center">
    <a href="https://arxiv.org/abs/2503.12035"><img src="https://img.shields.io/badge/arXiv-2503.12035-b31b1b"></a>
    <a href="https://github.com/CVMI-Lab/SlotCon/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
</p>

Welcome to the official repository for the  [MOS: Modeling Object-Scene Associations in Generalized Category Discovery](https://arxiv.org/abs/2503.12035) project!

## Running

### Dependencies

```
pip install -r requirements.txt
```

We recommend using the same configuration as ours: Python 3.8, CUDA > 12, and torch 2.3.1.

### Datasets

We use fine-grained benchmarks in this paper, including:

* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

In addition, we need to extract the mask for each image (where pixel value 255 represents the object and 0 represents the scene). Please follow the [IS-Net](https://github.com/danielgatis/rembg) for this process (model is isnet-general-use). Alternatively, you can use the pre-processed masks that we have already prepared. The Google Drive link is [link](https://drive.google.com/drive/folders/1sNVgc-iuxMTM7iP5tCppTwMrXbTObxq3?usp=drive_link).

The placement of the mask foler is as follows:

- For **cub**: your_path/cub/masks
- For **stanford_car**: your_path/stanford_car/cars_train_mask and your_path/stanford_car/cars_test_mask
- For **aircraft**: your_path/fgvc-aircraft-2013b/data/masks
- For **oxford-pet**: your_path/Oxford-pet/data/masks

### Scripts

**Train the model**:

```
bash scripts/run_${DATASET_NAME}.sh
```
Please note that in the `.sh` file, you need to specify the root directory of the dataset and DINO weight.

### Checkpoints

You can contact pengzhengyuan@sjtu.edu.cn to obtain logs and checkpoints from multiple experiments for any dataset. Feel free to reach out.

## Note

Please note that we have commented out the last norm layer in the DINO backbone.

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
