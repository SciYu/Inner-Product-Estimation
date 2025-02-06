# Inner Product Matrix Estimation for Incomplete Data

The source code for the WWW'2025 paper, titled **"A Theory-Driven Approach to Inner Product Matrix Estimation
for Incomplete Data: An Eigenvalue Perspective"**.

## Introduction

We provide a demo of similarity search on incomplete data, as detailed in Sections 6.2 and 6.3.

The code has been tested on MATLAB R2020b and R2024a. It should be able to run on other recent versions.

## Folders and Files

<pre>
./                              - Top directory.
./README.md                     - This readme file.
./experiments.m                 - Demo of similarity search on incomplete data.

|Data/                          - Incomplete data
   ./CIFAR10_n1000_80miss.mat   - Incomplete CIFAR10 dataset with 1,000 incomplete samples and 80% random missing

|Imputation/                    - Baselines of data imputation
   ./impute_mean.m              - Mean Imputation
   ./impute_knn.m               - kNN Imputation
   ./impute_svt.m               - SVT matrix completion
   ./impute_kfmc.m              - KFMC matrix completion
   ./impute_pmc.m               - PMC matrix completion

|Calibration/                   - Baselines of similarity calibration
   ./calibrate_dmc.m            - DMC calibration method  
   ./calibrate_smc.m            - SMC calibration method (see Algorithm 2 in the reference)
   ./calibrate_svc.m            - SVC calibration method (see Algorithm 2 in the reference)

|Our_method/                    - Our proposed eigenvalue correction algorithms 
   ./correct_ec_iid.m           - Eigenvalue correction for i.i.d. data (see Algorithm 1 in Section 3.3)
   ./correct_ec_noniid.m        - Eigenvalue correction for non-i.i.d. data (see Algorithm 2 in Section 4.3) 
   ./correct_ec_scale.m         - Scalable eigenvalue correction algorithm (see Algorithm 3 in Section 5.1 and Appendix B)

|Utils/                         - Evaluation files 
   ./similarity.m               - Inner product estimation on incomplete data
   ./eval_recall.m              - Measure Recall for similarity search tasks
</pre>

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{yu2025theory,
  title={A Theory-Driven Approach to Inner Product Matrix Estimation for Incomplete Data: An Eigenvalue Perspective},
  author={Yu, Fangchen and Zeng, Yicheng and Mao, Jianfeng and Li, Wenye},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  year={2025}
}
```

## Contact

If you have any problems or questions, please contact the author: Fangchen Yu (email: fangchenyu@link.cuhk.edu.cn)