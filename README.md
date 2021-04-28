# Learning Bloch Simulations for MR Fingerprinting by Invertible Neural Networks
This repository contains code for the [MLMIR 2020](https://sites.google.com/view/mlmir2020/) paper "Learning Bloch Simulations for MR Fingerprinting by Invertible Neural Networks", which can be found at [https://doi.org/10.1007/978-3-030-61598-7_6](https://doi.org/10.1007/978-3-030-61598-7_6).

## Installation

The installation has been tested with Ubuntu 18.04, Python 3.6, and PyTorch 1.4.0 with CUDA 10.1. The ``requirements.txt`` file lists all dependencies.

First, create a virtual environment named `innmrf` with Python 3.6:

    $ virtualenv --python=python3.6 innmrf
    $ source ./innmrf/bin/activate

Second, copy the code:

    $ git clone https://github.com/fabianbalsiger/mrf-reconstruction-mlmir2020
    $ cd mrf-reconstruction-mlmir2020

Third, install the required libraries:

    $ pip install -r requirements.txt

## Dataset

We provide small dictionaries for prototyping. The larger dictionaries used in the paper are available upon reasonable request.
The prototyping training dictionary was simulated with:

```
FF = 0:0.1:1;
T1H2O = [500:50:1500, 1550:100:2300];
T1fat = 325:25:475;
Df = -60:20:60;
B1 = 0.5:0.1:1;
```

The prototyping validation dictionary was simulated with:

```
FF = 0.05:0.1:1;
T1H2O = [525:100:1525, 1575:200:2300];
T1fat = 325:50:475;
Df = -60:30:60;
B1 = 0.55:0.15:1;
```

Both dictionaries can be downloaded by executing the script

    $ cd ./scripts/data
    $ python ./pull_example_data.py

The dictionaries can then be found in the directory ``./in``.

## Scripts
### Training the INN
To train the INN, simply execute ``training.py`` in the ``bin`` directory. The data and training parameters are provided by the ``./bin/config/config.json``, which you can adapt to your needs.
Note that you might want to specify the CUDA device and Python path by

    $ CUDA_VISIBLE_DEVICES=0 PYTHONPATH=".." python ./training.py

The script will automatically use the prototyping dictionaries in the directory ``./in/small``.
The training and validation will be logged under the path ``train_dir`` specified in the configuration file ``config.yaml``, which is by default the directory ``./out``.
To visualize the training process, we are using [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard).
Start the TensorBoard to observe the training by

    $ tensorboard --logdir=<path to the train_dir>

Note that the configuration file ``invfwdbwd_lr=1e-4_bs=50.yaml`` of the INN for the publication can be found in the ``configs_final`` directory.

### Training the baselines
The baselines are trained using the ``training_baseline.py`` script and the configuration file ``config_baseline.yaml``. Use the ``model`` property in the configuration file to choose the baseline.

        $ CUDA_VISIBLE_DEVICES=0 PYTHONPATH=".." python ./training_baseline.py

The configuration files for all baselines can be found in the ``configs_final`` directory.

### Testing
For testing, an universal script ``testing.py`` for both the INN and the baselines exists.

        $ CUDA_VISIBLE_DEVICES=0 PYTHONPATH=".." python ./testing.py

An example testing configuration file can be found at ``./bin/config/config_test.yaml`` directory. Note that the ``model_path`` needs to be a path to a checkpoint. 

### Plots and Evaluation
Executing the testing script will save multiple files, which can be used to reproduce the plots and the table with the results. The directory ``scripts`` contains all necessary plotting scripts. Pay attention to the script arguments!

## Models
The ``out`` directory contains trained models for the INN and all baselines as well as the TensorBoard logging. They were trained on the small dictionary. All paths in the testing configration files and the plotting scripts are set such that these trained models are used and the code can be executed without modifications. Further, using the training scripts and the provided training configuration files should yield exactly the same models (and, therefore, also results and plots).   

## Support
We leave the explanation of the code as exercise. But if you found a bug or have a specific question, please open an issue or a pull request. And please, try to figure out problems by yourself first using Google, Stack Overflow, or similar!

## Citation

If you use this work, please cite

```
Balsiger, F., Jungo, A., Scheidegger, O., Marty, B., & Reyes, M. (2020). Learning Bloch Simulations for MR Fingerprinting by Invertible Neural Networks. Machine Learning for Medical Image Reconstruction.
```

```
@inproceedings{BalsigerJungo2020,
archivePrefix = {arXiv},
arxivId = {2008.04139},
address = {Cham},
author = {Balsiger, Fabian and Jungo, Alain and Scheidegger, Olivier and Marty, Benjamin and Reyes, Mauricio},
booktitle = {Machine Learning for Medical Image Reconstruction},
doi = {10.1007/978-3-030-61598-7_6},
editor = {Deeba, Farah and Johnson, Patricia and W{\"{u}}rfl, Tobias and Ye, Jong Chul},
pages = {60--69},
publisher = {Springer},
series = {Lecture Notes in Computer Science},
title = {{Learning Bloch Simulations for MR Fingerprinting by Invertible Neural Networks}},
url = {http://arxiv.org/abs/2008.04139},
volume = {12450},
year = {2020}
}
```

## License

The code is published under the [MIT License](https://github.com/fabianbalsiger/mrf-reconstruction-mlmir2020/blob/master/LICENSE).
