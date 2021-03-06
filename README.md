# Overview

Modified version of the original repository [here](https://github.com/chaneyddtt/Generating-Multiple-Hypotheses-for-3D-Human-Pose-Estimation-with-Mixture-Density-Network). Cleaned and edited to do cross-dataset analysis.

In the original paper, they used H36M's 2D basic 16 joints configuration (including root joint) to train, but for 3d they removed the root and replaced it with the noise joint. Meaning they were using the following indices for 2D joints: 

```
[0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]
```
and for 3d:
```
[1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
```

For GPA, the original joint ordering appears to be flipped from the original one.

# Datasets
Datasets can be found [here](https://drive.google.com/drive/folders/1AeHrozrHHPUuDdW4Shm6QX-K12K566c4?usp=sharing). Even though, the datasets are here, you will need proper authorization to use in research.

# Quantative Results

## 2D Centered
| Train(&#8595;) Test(&#8594;)   |  H36M  |   GPA  |  3DPW  | 3DPW_AUG | SURREAL |    CKPT_NAME        | CKPT_INDEX |
|--------------------------------|--------|--------|--------|----------|---------|---------------------|------------|
| H36M                           |  49.81 |  86.76 |  90.84 |  89.55   |   85.25 | h36m_cnt2d_1        | 24371      |
| GPA                            |  93.73 |  52.16 | 106.30 | 106.21   |   99.29 | gpa_cnt2d_1         | 34760      |
| 3DPW                           | 115.82 | 106.01 |  70.26 |  79.34   |   97.34 | 3dpw_cnt2d_1        |  7810      |
| 3DPW_AUG                       | 111.61 | 103.23 |  72.57 |  61.82   |   96.09 | 3dpw_aug_cnt2d_1    | 12432      |
| SURREAL                        |        |        |        |          |         |                     |            |
| H36M (ELEV AUG x2)             |  49.15 |  83.39 |  86.86 |  84.16   |   88.81 | h36m_elvaug_cnt2d_2 | 36556      |
| H36M (ELEV AUG x3)             |  48.72 |  84.43 |  90.48 |  86.69   |   90.43 | h36m_elvaug_cnt2d_1 | 42649      |

## Without 2D Centered

| Train(&#8595;) Test(&#8594;)   |  H36M  |  GPA   |  3DPW  | 3DPW_AUG | SURREAL |
|--------------------------------|--------|--------|--------|----------|---------|
| H36M                           |  61.46 | 140.95 |  97.66 |   152.53 |  156.63 |
| GPA                            | 135.37 |  70.23 | 130.03 |   188.21 |  179.80 |
| 3DPW                           | 125.65 | 128.83 |  67.15 |   142.13 |  131.78 |
| 3DPW_AUG                       | 162.54 | 154.22 |  95.88 |    78.64 |  118.95 |
| SURREAL                        |        |        |        |          |   65.21 |
| H36M (ELEV AUG x2)             |  69.86 | 136.60 | 122.47 |          |         |

## Command Line Args
```
--train_dir: Training will checkpoint initially at "experiments/${train_dir}"
--load_dir : Model will load and save later training progress at "experiments/${load_dir}"
--load     : checkpoint number (must if you specified load_dir)
--test     : True if you want to evaluate
--dataset  : Supported dataset: h36m, gpa, 3dpw, 3dpw_aug
--qual     : Test and save images to "out/${qual}"
```

# Qualitative Results
First 5 of each row are different hypothesis. And the last column is the ground truth.

## Train: 3DPW Test: 3DPW
| H1 | H2 | H3 | H4 | H5 | GT |
|----|----|----|----|----|----|
| <img src="./out/3dpw_3dpw/0_0.png"> | <img src="./out/3dpw_3dpw/0_1.png"> | <img src="./out/3dpw_3dpw/0_2.png"> | <img src="./out/3dpw_3dpw/0_3.png"> | <img src="./out/3dpw_3dpw/0_4.png"> | <img src="./out/3dpw_3dpw/0_gt.png"> |
| <img src="./out/3dpw_3dpw/1_0.png"> | <img src="./out/3dpw_3dpw/1_1.png"> | <img src="./out/3dpw_3dpw/1_2.png"> | <img src="./out/3dpw_3dpw/1_3.png"> | <img src="./out/3dpw_3dpw/1_4.png"> | <img src="./out/3dpw_3dpw/1_gt.png"> |
| <img src="./out/3dpw_3dpw/3_0.png"> | <img src="./out/3dpw_3dpw/3_1.png"> | <img src="./out/3dpw_3dpw/3_2.png"> | <img src="./out/3dpw_3dpw/3_3.png"> | <img src="./out/3dpw_3dpw/3_4.png"> | <img src="./out/3dpw_3dpw/3_gt.png"> |
| <img src="./out/3dpw_3dpw/10_0.png"> | <img src="./out/3dpw_3dpw/10_1.png"> | <img src="./out/3dpw_3dpw/10_2.png"> | <img src="./out/3dpw_3dpw/10_3.png"> | <img src="./out/3dpw_3dpw/10_4.png"> | <img src="./out/3dpw_3dpw/10_gt.png"> |

## Train: H36M Test: 3DPW
| H1 | H2 | H3 | H4 | H5 | GT |
|----|----|----|----|----|----|
| <img src="./out/h36m_3dpw/0_0.png"> | <img src="./out/h36m_3dpw/0_1.png"> | <img src="./out/h36m_3dpw/0_2.png"> | <img src="./out/h36m_3dpw/0_3.png"> | <img src="./out/h36m_3dpw/0_4.png"> | <img src="./out/h36m_3dpw/0_gt.png"> |
| <img src="./out/h36m_3dpw/1_0.png"> |  <img src="./out/h36m_3dpw/1_1.png"> | <img src="./out/h36m_3dpw/1_2.png"> | <img src="./out/h36m_3dpw/1_3.png"> | <img src="./out/h36m_3dpw/1_4.png"> | <img src="./out/h36m_3dpw/1_gt.png"> |
| <img src="./out/h36m_3dpw/9_0.png"> |  <img src="./out/h36m_3dpw/9_1.png"> | <img src="./out/h36m_3dpw/9_2.png"> | <img src="./out/h36m_3dpw/9_3.png"> | <img src="./out/h36m_3dpw/9_4.png"> | <img src="./out/h36m_3dpw/9_gt.png"> |
| <img src="./out/h36m_3dpw/10_0.png"> | <img src="./out/h36m_3dpw/10_1.png"> | <img src="./out/h36m_3dpw/10_2.png"> | <img src="./out/h36m_3dpw/10_3.png"> | <img src="./out/h36m_3dpw/10_4.png"> | <img src="./out/h36m_3dpw/10_gt.png"> |
| <img src="./out/h36m_3dpw/11_0.png"> | <img src="./out/h36m_3dpw/11_1.png"> | <img src="./out/h36m_3dpw/11_2.png"> | <img src="./out/h36m_3dpw/11_3.png"> | <img src="./out/h36m_3dpw/11_4.png"> | <img src="./out/h36m_3dpw/11_gt.png"> |

## Train: GPA Test: 3DPW
| H1 | H2 | H3 | H4 | H5 | GT |
|----|----|----|----|----|----|
| <img src="./out/gpa_3dpw/0_0.png"> | <img src="./out/gpa_3dpw/0_1.png"> | <img src="./out/gpa_3dpw/0_2.png"> | <img src="./out/gpa_3dpw/0_3.png"> | <img src="./out/gpa_3dpw/0_4.png"> | <img src="./out/gpa_3dpw/0_gt.png"> |
| <img src="./out/gpa_3dpw/1_0.png"> | <img src="./out/gpa_3dpw/1_1.png"> | <img src="./out/gpa_3dpw/1_2.png"> | <img src="./out/gpa_3dpw/1_3.png"> | <img src="./out/gpa_3dpw/1_4.png"> | <img src="./out/gpa_3dpw/1_gt.png"> |
| <img src="./out/gpa_3dpw/2_0.png"> | <img src="./out/gpa_3dpw/2_1.png"> | <img src="./out/gpa_3dpw/2_2.png"> | <img src="./out/gpa_3dpw/2_3.png"> | <img src="./out/gpa_3dpw/2_4.png"> | <img src="./out/gpa_3dpw/2_gt.png"> |
| <img src="./out/gpa_3dpw/7_0.png"> | <img src="./out/gpa_3dpw/7_1.png"> | <img src="./out/gpa_3dpw/7_2.png"> | <img src="./out/gpa_3dpw/7_3.png"> | <img src="./out/gpa_3dpw/7_4.png"> | <img src="./out/gpa_3dpw/7_gt.png"> |
| <img src="./out/gpa_3dpw/12_0.png"> | <img src="./out/gpa_3dpw/12_1.png"> | <img src="./out/gpa_3dpw/12_2.png"> | <img src="./out/gpa_3dpw/12_3.png"> | <img src="./out/gpa_3dpw/12_4.png"> | <img src="./out/gpa_3dpw/12_gt.png"> |

## Train: GPA Test: GPA
| H1 | H2 | H3 | H4 | H5 | GT |
|----|----|----|----|----|----|
| <img src="./out/gpa_gpa/0_0.png"> | <img src="./out/gpa_gpa/0_1.png"> | <img src="./out/gpa_gpa/0_2.png"> | <img src="./out/gpa_gpa/0_3.png"> | <img src="./out/gpa_gpa/0_4.png"> | <img src="./out/gpa_gpa/0_gt.png"> |
| <img src="./out/gpa_gpa/1_0.png"> | <img src="./out/gpa_gpa/1_1.png"> | <img src="./out/gpa_gpa/1_2.png"> | <img src="./out/gpa_gpa/1_3.png"> | <img src="./out/gpa_gpa/1_4.png"> | <img src="./out/gpa_gpa/1_gt.png"> |
| <img src="./out/gpa_gpa/2_0.png"> | <img src="./out/gpa_gpa/2_1.png"> | <img src="./out/gpa_gpa/2_2.png"> | <img src="./out/gpa_gpa/2_3.png"> | <img src="./out/gpa_gpa/2_4.png"> | <img src="./out/gpa_gpa/2_gt.png"> |
|  <img src="./out/gpa_gpa/3_0.png"> | <img src="./out/gpa_gpa/3_1.png"> | <img src="./out/gpa_gpa/3_2.png"> | <img src="./out/gpa_gpa/3_3.png"> | <img src="./out/gpa_gpa/3_4.png"> | <img src="./out/gpa_gpa/3_gt.png"> |
| <img src="./out/gpa_gpa/4_0.png"> | <img src="./out/gpa_gpa/4_1.png"> | <img src="./out/gpa_gpa/4_2.png"> | <img src="./out/gpa_gpa/4_3.png"> | <img src="./out/gpa_gpa/4_4.png"> | <img src="./out/gpa_gpa/4_gt.png"> |

## Train: GPA Test: H36M
| H1 | H2 | H3 | H4 | H5 | GT |
|----|----|----|----|----|----|
| <img src="./out/gpa_h36m/0_0.png"> | <img src="./out/gpa_h36m/0_1.png"> | <img src="./out/gpa_h36m/0_2.png"> | <img src="./out/gpa_h36m/0_3.png"> | <img src="./out/gpa_h36m/0_4.png"> | <img src="./out/gpa_h36m/0_gt.png"> |
| <img src="./out/gpa_h36m/1_0.png"> | <img src="./out/gpa_h36m/1_1.png"> | <img src="./out/gpa_h36m/1_2.png"> | <img src="./out/gpa_h36m/1_3.png"> | <img src="./out/gpa_h36m/1_4.png"> | <img src="./out/gpa_h36m/1_gt.png"> |
| <img src="./out/gpa_h36m/2_0.png"> | <img src="./out/gpa_h36m/2_1.png"> | <img src="./out/gpa_h36m/2_2.png"> | <img src="./out/gpa_h36m/2_3.png"> | <img src="./out/gpa_h36m/2_4.png"> | <img src="./out/gpa_h36m/2_gt.png"> |
| <img src="./out/gpa_h36m/3_0.png"> | <img src="./out/gpa_h36m/3_1.png"> | <img src="./out/gpa_h36m/3_2.png"> | <img src="./out/gpa_h36m/3_3.png"> | <img src="./out/gpa_h36m/3_4.png"> | <img src="./out/gpa_h36m/3_gt.png"> |
| <img src="./out/gpa_h36m/4_0.png"> | <img src="./out/gpa_h36m/4_1.png"> | <img src="./out/gpa_h36m/4_2.png"> | <img src="./out/gpa_h36m/4_3.png"> | <img src="./out/gpa_h36m/4_4.png"> | <img src="./out/gpa_h36m/4_gt.png"> |


| GPA_FLIPPED                    | 196.20 |        | 164.10 |   176.34 |       72.08 |