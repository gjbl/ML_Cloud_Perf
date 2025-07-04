# ML_Cloud_Perf
Performance evaluation of ML Algorithms in GC and AWS cloud environments

This is code and data suporting the paper "Performance evaluation of ML Algorithms in GC and AWS cloud environments" by Grzegorz Blinowski1[0000-0002-0869-2828] and Marcin Bogucki2 to appear in COMSYS2025 conference

Paper abstract:
 In this paper, we analyze the performance of common machine learning (ML) algorithms executed in Google Cloud (GC) and Amazon Web Services (AWS) environments. The primary metric is training and prediction time as a function of the number of virtual machine (VM) cores. For com-parison, benchmarks also include a "bare metal" (i.e., non-cloud) environ-ment, with results adjusted using the "Multi-thread Score" to account for ar-chitectural differences among the tested platforms.
Our focus is on CPU-intensive algorithms. The test suite includes Support Vector Machines, Decision Trees, K-Nearest Neighbors (KNN), Linear Mod-els, and Ensemble Methods. The evaluated classifiers, sourced from the scikit-learn and ThunderSVM libraries, include: Extra Trees, Support Vector Machines, K-Nearest Neighbors, Random Forest, Gradient Boosting Classifi-er, and Stochastic Gradient Descent (SGD). GPU-accelerated deep learning models, such as large language models (LLMs), are excluded due to the dif-ficulty of establishing a common baseline across platforms.
The dataset used is the widely known "Higgs dataset," which describes kinematic properties measured by particle detectors in the search for the Higgs boson. It contains 940,160 instances, 25 features, and 2 solution clas-ses. The dataset is balanced, meaning both classes contain an equal number of instances.
The dataset chosen is a popular "Higgs set" (which describes kinematic properties measured by the particle detectors in accelerator in the search of Higgs boson. It contains: 940160 instances, 25 features and 2 solution clas-ses. The dataset is balanced, meaning the classes has the same number of el-ements for both solution classes.
Benchmark results are best described as varied—there is no clear trend, as training and prediction times scale differently depending on both the cloud platform and the algorithm type. This paper provides practical insights and guidance for deploying and optimizing CPU-based ML workloads in cloud environments.
Keywords: Public Cloud, ML/AI algorithms, performance evaluation.
