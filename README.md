# Skew-Normal VAEs: Learning Asymmetric Motion Distributions for Trajectory Prediction from the Latent Space (Under Review)

This repository provides the code for [skew-normal variational autoencoders (SNVAEs)](./skew_normal_class.py). The SNVAE is applied to a trajectory prediction method [GroupNet](https://github.com/MediaBrain-SJTU/GroupNet). The resulting method is named SN-GroupNet.

Here are the scripts for training and testing SN-GroupNet.

Please first download the data files `train.npy` and `test.npy` provided by the [GroupNet](https://github.com/MediaBrain-SJTU/GroupNet), and place them in the path `./datasets/nba`. Then, run experiments by using the following commands.

```bash
# Training
# SN-GroupNet
python train_hyper_nba.py --ztype top1skew --learn_prior
# SN-GroupNet w/o SP
python train_hyper_nba.py --ztype skew --learn_prior
# SN-GroupNet-Prior
python train_hyper_nba.py --ztype top1skew
# SN-GroupNet-Prior w/o SP
python train_hyper_nba.py --ztype skew

# Testing
# Replace "YYYYMMDD-HHMMSS" the date of your experiment
python test_nba.py --model_names model --dist top1skew --date "YYYYMMDD-HHMMSS"
```
