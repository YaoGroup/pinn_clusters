[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YaoGroup/pinn_clusters/blob/main/example.ipynb)

# pinn_clusters
Accompanying code and results for the paper: "[1D Ice Shelf Hardness Inversion: Clustering Behavior and Collocation Resampling in
Physics-Informed Neural Networks](https://doi.org/10.1016/j.jcp.2023.112435)"  by Yunona Iwasaki and Ching-Yao Lai. We document the codes used to train PINNs for 1D ice-shelf inverse modeling. Additionally,  we provide scripts to facilitate analysis of training results over repeated trials.

The software used in the example trial are pulled from the GitHub repository [YaoGroup/ice_1D_pinn](https://github.com/YaoGroup/ice_1D_pinn/tree/main). A detailed documentation and description of the code mechanism can be found at this link. To conduct more realistic studies involving multiple parameter values and many trials, we recommend directly installing the software package onto your device following the installation instructions in this software repo.

Please direct questions about this code and documentation to Ching-Yao Lai (cyaolai@stanford.edu) and Yunona Iwasaki (yiwasaki@berkeley.edu).
## Overview
In our study, we investigated the feasibility of a physics-informed neural network to invert for the correct ice-shelf hardness spatial profile $B(x)$ given sparse, noisy training data of only velocity $u(x)$ and thickness $h(x)$. We found that for training data with realistic levels of noise, PINNs trained using the original formulation of the objective function (equal weighting of equation and data loss) perform poorly over a broad range of PINN hyperparameters. 

Thus, we introduced a weighting hyperparameter $\gamma$ in the objective function of the PINN, adjusting the relative weighting between equation and data loss. $\frac{\gamma}{1-\gamma} = 1$ corresponds to the original formulation in which the equation and data losses are weighted equally.  When $\frac{\gamma}{1-\gamma} > 1$, the equation loss is weighted more heavily than the data loss, and vice versa for  $\frac{\gamma}{1-\gamma} < 1$. We predicted that by weighting the equation loss heavier than the data loss by setting $\frac{\gamma}{1-\gamma} > 1$, PINNs would be better able to denoise the training data and attain higher predictive accuracies.

Indeed, we observed that the _attainable_ predictive accuracy of PINNs improves significantly for higher values of $\gamma$. __However, we also discover that for $\frac{\gamma}{1-\gamma} \gtrapprox 1$, PINN predictive performance is highly unreliable.__ The prediction errors are distributed bimodally, with one cluster corresponding to accurate, "low-error" solutions, and the other consisting of "high-error" solutions that fit poorly to the ground truth datasets. We call this phenomenon the _clustering behavior_ (see Figure 1). This behavior persisted over an extensive range of hyperparameters (see paper).

![Highlight clustering for default training scheme](https://github.com/YaoGroup/pinn_clusters/blob/main/images/clusterplot_forgithub.png)

<p align="center">
Figure 1. The clustering behaviour of  PINN predictions. Correlation of $B_{err}$ with $u_{err}$ and $h_{err}$ over 501 trials for various values of $\frac{\gamma}{1-\gamma}$ and noise level = 0.3, using six, 20-unit hidden layers and $c=1001$ collocation points. PINNs were trained for 400,000 iterations of Adam and 200,000 iterations of LBFGS, or until convergence. (a),(b): $u_{err}$ and $h_{err}$ vs. $B_{err}$, respectively.
</p>

Given this problematic behavior, we conclude that setting $\frac{\gamma}{1-\gamma} > 1$ is not a viable method for improving prediction accuracy. Thus, we tested an alternative approach to improving PINN performance accuracy which we call _collocation resampling_, in which the collocation points used to evaluate the equation loss are resampled after every training iteration. Using this approach, clustering was again exhibited for values of $\frac{\gamma}{1-\gamma} \gtrapprox 1$. However, we discover that for $\frac{\gamma}{1-\gamma} \lessapprox 1$, we observe a 2-3 orders of magnitude decrease in predictive errors compared to training with fixed collocation points, with no clustering. We present these results in Figure 2.

![Highlight clustering for random collocation resampling](https://github.com/YaoGroup/pinn_clusters/blob/main/images/collocation_resampling_highlight.png)
<p align="center">
Figure 2. Distribution of $B_{err}$, $u_{err}$, and $h_{err}$ using collocation resampling for constant $B(x)$ profile over 501 trials with noise = 0.3. We compare the trial error distributions for values of  $\frac{\gamma}{1-\gamma} < 1.$ No clustering is observed for these values of $\gamma$ using initial training with collocation resampling. In grey, we overlay the prediction errors from all PINN solutions obtained with $\frac{\gamma}{1-\gamma} < 1$ using the original training scheme.
</p>

# Table of Contents
## example.ipynb
Main script for training PINNs to predict for the correct 1D $u(x)$ (velocity), $h(x)$ (thickness), and $B(x)$ (hardness) profiles given synthetic noisy data for $u(x)$ and $h(x)$. In addition to training PINNs, this script handles generation of synthetic noisy training data at a specified noise level, as well as evaluation of PINN predictive accuracy compared to ground truth profiles. 

This script requires the user to specify the ground truth $u(x)$, $h(x)$, and $B(x)$ profiles. Currently, $u(x)$ and $h(x)$ profiles are specified by providing a reference to a Python dictionary saved as a ```.mat``` file using the ```N_t``` variable. The ground truth $B(x)$ profile is specified by passing an array of values to the ```B_truth``` variable. ```B_truth``` should contain the ground truth values of $B(x)$ at the same values of $x$ as the ground truth $u(x)$ and $h(x)$ profiles.

Additionally, this script requires users to specify the following hyperparameters relevant to our optimization study:

* ```N_t``` _(int)_: Number of collocation points. This number stays fixed, even if the script switches between collocation resampling and fixed collocation points.
* ```N_ob``` _(int)_: Number of observation points.
* ```layers``` _(list)_: List specifying the width and depth of the neural network. Specify the size of each layer except for the input layer. e.g. ```layers = [5,5,3]``` for a neural network with two, 5-unit hidden layers. The final value specifies the size of the output layer and must be set to 3 for this problem.

* ```num_iterations_adam_resampled```, ```num_iterations_adam_fixed```, ```num_iterations_lbfgs``` _(int)_: Specify the number of iterations to train with each optimizer and collocation method.
  * ```adam_resampled```: train with Adam optimizer using collocation resampling.
  * ```adam_fixed```: train with Adam optimizer wih fixed collocation points
  * ```lbfgs```: train with L-BFGS optimizer with fixed collocation points.
    
Thus, to train *only* with fixed collocation points, please set 

```num_iterations_adam_resampled = 0``` 

Conversely, to train *only* with collocation resampling, please set 

```num_iterations_adam_fixed = 0```

```num_iterations_lbfgs = 0```

   *Note: it does not make sense to have a* ```num_iterations_lbfgs_resampled``` *parameter, because L-BFGS is a second order method in which the update to the neural network weights is informed by the Hessian, which is estimated based on the evaluation of gradients over the past several iterations. Thus L-BFGS is incompatible with the collocation resampling method described in our paper. Resampling the collocation points after every iteration will cause training to terminate prematurely.*

* ```test_noise``` _(float)_: level of noise added to ground truth $u(x)$ and $h(x)$ profiles during synthetic data generation. Please refer to p. 6 of the main text for the definition of noise level; it may also be helpful to see its implementation in the script ```noise.py```.

* ```test_gammas``` _(list)_: specify one or multiple values of $\gamma$ to test. To conveniently implement logarithmic spacing of $\frac{\gamma}{1-\gamma}$, the user may the specify $\log_{10}(\frac{\gamma}{1-\gamma})$ values using the ```logratios``` variable, then solve for the corresponding $\gamma$-values in the next line.

## /data

Includes ground truth profiles for $u(x)$ and $h(x)$ from which noisy data is generated. Both ground truth profiles assume boundary conditions $u(0) = 1$, $h(0) = h_0$. See p. 5 of the main text and pp. 2-3 of the supplementary material for the definitions and numerical values of $h_0$ and other relevant constants. 


### ```constantB_uh.mat```
Analytic $u(x)$ and $h(x)$ solutions for $B(x) = 1.0$, uniformly sampled for $x \in [0.0,1.0]$. 

### ```sinusoidalB_uh.mat```
Numerical $u(x)$ and $h(x)$ solutions for $B(x) = \\frac{1}{2} \cos{(3\pi x)}$, uniformly sampled for $x \in [0.0,1.0]$. 


## /trial

This folder contains all trial results discussed in our paper as well as scripts written to facilitate visualization and analysis of results.

### ```trial_processing.ipynb```
Jupyter notebook for consolidating error data from a set of trial result dictionaries into a single numpy array.

### ```pinn_cluster_plots.ipynb```
Jupyter notebook that loads the numpy error array of a set of training trials and separates trials by $k$-means clustering in log-space. Incllludes code for vizualising clusters, plotting cluster statistics, etc.

### /trial/trial_results
Contains numpy arrays of the $B_{\mathrm{err}}$, $u_{\mathrm{err}}$, $h_{\mathrm{err}}$ for different experiments studied in the paper. Each numpy array has shape $(n, m, l)$, where $n$ is the number of values of $\gamma$ tested in the experiment, $m = 3$ is the number of predictive variables (i.e. $u$, $h$, $B$), and $l$ is equal to the number of repeated trials. In this repo, $l=501$ for all experiments. Please use the following code to load each array:

```
errors = np.load('path_to_file/errors.npy')

gi = 1

u_errs = errors[gi][0]
h_errs = errors[gi][1]
B_errs = errors[gi][2]
```
where ```gi``` should be modified to the index of the $\gamma$-value being examined.

Naming conventions for each numpy array are as follows:

#### ```clean_u206l1kc_errs.npy```, ```nx_u206l1kc_errs.npy```
Error results using the standard settings(six, 20-unit hidden layers with $c=1001$ fixed collocation points). The prefix "clean" corresponds to tests using clean training data; prefixes "nx" denote tests using noisy training data, with x specifying the level of noise (i.e. n3_u206l1kc_errs.npy corresponds to noise = 0.3; n05_u206l1kc_errs.npy corresponds to noise = 0.05)

#### ```resampled_fixed.npy```
Results from training with collocation resampling followed by fixed collocation points for noise level $= 0.3$, (see Section 3.4 of the main text, pp. 14-15).
We tested $l = 12$ values of $\gamma$ such that $\frac{\gamma}{1-\gamma}$ are logarithmically spaced between $[10^{-4}, 10^6]$, i.e., ```gi = 0``` corresponds to $\frac{\gamma}{1-\gamma} = 10^{-4}$, ```gi = 1``` corresponds to $\frac{\gamma}{1-\gamma} = 10^{-3}$, ...```gi = 11``` corresponds to $\frac{\gamma}{1-\gamma} = 10^{7}$. Note that we omit the largest value of $\gamma$ tested in the experiments using fixed collocation points, so that we have one fewer value of $\gamma$. 


#### ```ux_errs.npy```
Results from tests with increased neural network width. ux corresponds to x-units per hidden layer. (Noise level = 0.3 for both)
#### ```u100_c1k_errs.npy```
Results from final test with two 100-hidden units and 1001 collocation points. (Noise level = 0.3)

## /images
Contains the images used in the README of this GitHub repo.

# Code Implementation of Collocation Resampling

Training can be switched between using fixed collocation points and collocation resampling by switching the loss function used during training. The loss function evaluated by a given optimizer is specified during the initialization of the optimizer. Use the  ```SquareLoss``` loss function when using fixed collocation points, and ```SquareLossRandom``` for random collocation resampling (see lines 77-100 in 'pinn_trials.py').

Comparing the ```SquareLoss``` and ```SquareLossRandom``` functions in 'loss.py', the main difference between the two functions is in the ```__call__``` method. For ```SquareLossRandom```, we add a few extra lines at the beginning of the  ```__call__``` method (lines 54-61):

```
def __call__(self, x_eqn, data_pts, net) -> Dict[str, tf.Tensor]:
    xmin = 0.0
    xmax = 1.0
    N_t = 1001
    _data_type = tf.float64       
    collocation_pts = xmin + (xmax - xmin) * self.col_gen.uniform(shape = [N_t])
    collocation_pts = collocation_pts**3
```
where ```self.col_gen``` is a stateful random generator defined in the ```__init__``` method (line 52):

```
        self.col_gen = tf.random.get_global_generator()
```
Thus, the ```SquareLossRandom``` function generates a new set of collocation points every time it is called, i.e. at every iteration. 

__Important Note: It is essential to use a _stateful_ random number generator such as ```tf.random.Generator()``` to ensure that the collocation points are resampled after each iteration.__ Using a stateless random generator (such as 
 those provided in the ```numpy.random``` module, or the ```lhs``` generator used in our codes for fixed collocation point generation) will not allow the collocation points to be updated in a TensorFlow training loop, causing the loss function to behave identically to training with fixed collocation points.

# Citation
Yunona Iwasaki and Ching-Yao Lai.
*One-dimensional ice shelf hardness inversion: Clustering behavior and collocation resampling in physics-informed neural networks.* Journal of Computational Physics, Volume 492, 2023, 112435, ISSN 0021-9991, https://doi.org/10.1016/j.jcp.2023.112435.

**BibTex:**
```
@article{IWASAKI2023112435,
          title = {One-dimensional ice shelf hardness inversion: Clustering behavior and collocation resampling in physics-            informed neural networks},
          journal = {Journal of Computational Physics},
          volume = {492},
          pages = {112435},
          year = {2023},
          issn = {0021-9991},
          doi = {https://doi.org/10.1016/j.jcp.2023.112435},
          url = {https://www.sciencedirect.com/science/article/pii/S0021999123005302},
          author = {Yunona Iwasaki and Ching-Yao Lai},
          keywords = {Physics-informed neural networks, Scientific machine learning, Ice dynamics, Geophysical fluid                   dynamics, Nonlinear dynamics, Inverse problems},
      }
```

