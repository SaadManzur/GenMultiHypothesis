Modified version of the original repository [here]{https://github.com/chaneyddtt/Generating-Multiple-Hypotheses-for-3D-Human-Pose-Estimation-with-Mixture-Density-Network}. Cleaned and edited to do cross-dataset analysis.

In the original paper, they used H36M's 2D basic 16 joints configuration (including root joint) to train, but for 3d they removed the root and replaced it with the noise joint. Meaning they were using the following indices for 2D joints: 

```
[0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]
```
and for 3d:
```
[1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
```

| Train(&#8595;) Test(&#8594;)   |  H36M  |  GPA   |  3DPW  |
|--------------------------------|--------|--------|--------|
| H36M                           |  61.46 | 140.95 |  97.66 |
| GPA                            | 135.37 |  70.23 | 130.03 |
| 3DPW                           | 125.65 | 128.83 |  67.15 |