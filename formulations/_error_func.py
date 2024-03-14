import time
import numpy as np 
import tensorflow as tf

from pinn_clusters import (create_mlp, LBFGS, Adam,random_sample, add_noise)
from ._formulations import _data_type
from ._formulations import (inverse_1st_order_equations, to_tensor, get_collocation_points, data_equations)
from ._loss import (SquareLossRandom, SquareLoss)


def initialize_model(layers,lyscl):
    """
    Initializes the neural network model.

    Args:
        layers (list): Network layer configuration.
        lyscl (list): Standard deviation to set the scales for Xavier weight initialization

    Returns:
        model: Initialized neural network model.
    """
    model = create_mlp(layers,lyscl)
    return model

def generate_synthetic_data(N_ob, x_star, u_star, h_star, noise_level):
    """
    Generates synthetic training data by adding noise to the ground truth data.

    Args:
        N_ob (int): Number of training points.
        x_star (np.array): X-locations of ground truth data.
        u_star (np.array): Ground truth u data.
        h_star (np.array): Ground truth h data.
        noise_level (float): Noise level to be added to the data.

    Returns:
        tuple: Tuple containing sampled x, u, and h data with added noise.
    """
    x_sampled, u_sampled, h_sampled = random_sample(
        N_ob, x_star,
        add_noise(u_star, ratio=noise_level),
        add_noise(h_star, ratio=noise_level)
    )
    return x_sampled, u_sampled, h_sampled

def train_model(model, equations, x_sampled, u_sampled, h_sampled, gamma, fractional, 
                x_star, N_t, num_iterations_adam_resampled,
                num_iterations_adam_fixed,
                num_iterations_lbfgs):
    """
    Trains the model using various optimization techniques.

    Args:
        model: The neural network model to be trained.
        equations: Equation we are simulating
        x_sampled, u_sampled, h_sampled: Sampled training data.
        gamma (float): The gamma value used in training.
        fractional (bool): Hyper parameters for the PINN
        x_star (nd.array): X location
        N_t (int): Number of collocation points
        num_iterations_adam_resampled (int) :umber of iterations of Adam using collocation resampling
        num_iterations_adam_fixed  (int): Number of iterations of Adam with fixed collocation points
        num_iterations_lbfgs (int): Number of iterations of LBFGS using fixed collocation points  
    Returns:
        dict: Dictionary containing training records and time elapsed.
    """
    loss_colo       = SquareLossRandom(equations=equations, equations_data=data_equations, gamma=gamma) #Initialize loss function for collocation resampled training

    collocation_pts = get_collocation_points(x_train=x_star, xmin=x_star.min(), xmax=x_star.max(), N_t=N_t) #Randomly sample a set of collocation points to be used for fixed collocation training
    loss            = SquareLoss(equations=equations, equations_data=data_equations, gamma=gamma) #Initialize loss function for fixed collocation training

    start_time = time.time()
    
    #Train using Adam with collocation resampling: initialize Adam optimizer with argument loss=loss_colo to enable collocation resampling.
    # NOTE: the Adam() initializer requires a "collocation_points" argument even when using collocation resampling. This was done for consistency with the fixed collocation version.
    # Whatever points are passed as argument are ignored when loss_colo is passed as the loss argument; in this case an empty array of collocation points can also be passed without error.
    
    adam_resampled = Adam(
        net=model, loss=loss_colo, collocation_points=collocation_pts,
        data_points=(to_tensor(x_sampled), (to_tensor(u_sampled), to_tensor(h_sampled)))
    )
    adam_resampled.optimize(nIter=num_iterations_adam_resampled)

    #Train using Adam with fixed collocation points: initialize Adam optimizer with argument loss=loss for fixed collocation points.
    adam_fixed = Adam(
        net=model, loss=loss, collocation_points=collocation_pts,
        data_points=(to_tensor(x_sampled), (to_tensor(u_sampled), to_tensor(h_sampled)))
    )
    adam_fixed.optimize(nIter=num_iterations_adam_fixed)

    #Train using LBFGS with fixed collocation points
    lbfgs = LBFGS(
        net=model, loss=loss, collocation_points=collocation_pts,
        data_points=(to_tensor(x_sampled), (to_tensor(u_sampled), to_tensor(h_sampled)))
    )
    lbfgs.optimize(nIter=num_iterations_lbfgs)
    elapsed = time.time() - start_time
    print(f'Training time: {elapsed:.4f} seconds')

    # [Code for compiling training records]
    equation_losses = np.array(adam_resampled.loss_records.get("loss_equation", [])+ (adam_fixed.loss_records.get("loss_equation", []) + lbfgs.loss_records.get("loss_equation", [])))
    data_losses = np.array(adam_resampled.loss_records.get("loss_data", []) + adam_fixed.loss_records.get("loss_data", []) + lbfgs.loss_records.get("loss_data", []))
    total_losses = np.array(adam_resampled.loss_records.get("loss", []) + adam_fixed.loss_records.get("loss", []) + lbfgs.loss_records.get("loss", []))

    data_losses = np.trim_zeros(data_losses, 'b')
    equation_losses = np.trim_zeros(equation_losses, 'b')
    total_losses = np.trim_zeros(total_losses, 'b')

    return {
       # "loss_records": loss_records,
       "data_losses": data_losses,
       "equation_losses": equation_losses,
       "total_losses": total_losses,
        "training_time": elapsed
    }

def process_results(model, equations, x_star, u_star, 
                    h_star,x_sampled,u_sampled,h_sampled,B_truth):
    """
    Processes the training results and calculates errors.

    Args:
        model: The trained model.
        equations: Equation we are simulating
        x_star, u_star, h_star: Ground truth data.
        x_sampled, u_sampled,h_sampled
        B_truth:nd_arrat B(x) profile used to solve for ground truth

    Returns:
        dict: Dictionary containing prediction errors and other analysis results.
    """

    x_star = tf.cast(x_star, dtype=_data_type)
    uhb_pred = model(x_star)
    f_pred = equations(x_star, model, drop_mass_balance=False)
    x_star = x_star.numpy().flatten().flatten()
    x_sampled = x_sampled.flatten().flatten()
    u_star = u_star.flatten()
    u_sampled = u_sampled.flatten()
    h_star = h_star.flatten()
    h_sampled = h_sampled.flatten()
    u_p = uhb_pred[:, 0:1].numpy().flatten()
    h_p = uhb_pred[:, 1:2].numpy().flatten()
    B_p = uhb_pred[:, 2:3].numpy().flatten()

    #compute B_err, u_err, h_err
    total_berr = err(B_p, B_truth)
    total_uerr = err(u_p, u_star)
    total_herr = err(h_p, h_star)

    return {
        "B_err": total_berr,
        "u_err" : total_uerr,
        "h_err" : total_herr,
        "B_p" : B_p, 
        "u_p" : u_p,
        "h_p" : h_p,
        "u_sampled" : u_sampled,
        "h_sampled" : h_sampled
    }       

def Berr_func(gamma, noise_level, x_star, u_star, h_star,
              layers,lyscl, N_ob, fractional,N_t, num_iterations_adam_resampled,
                num_iterations_adam_fixed,
                num_iterations_lbfgs, B_truth):
    """
    Function to generate synthetic training data, train PINNs, and analyze results.

    Args:
        gamma (float): Value of gamma used for training.
        noise_level (float): Level of noise added to synthetic training data.
        x_star (np.array): X-locations of ground truth u and h training data.
        u_star (np.array): Ground truth u data.
        h_star (np.array): Ground truth h data.
        layers (list): Network layer configuration.
        lyscl (list): Standard deviation to set the scales for Xavier weight initialization
        N_ob (int): Number of training points.
        fractional (bool): Hyper parameters for the PINN
        N_t (int): Number of collocation points
        num_iterations_adam_resampled (int) :umber of iterations of Adam using collocation resampling
        num_iterations_adam_fixed  (int): Number of iterations of Adam with fixed collocation points
        num_iterations_lbfgs (int): Number of iterations of LBFGS using fixed collocation points  
        B_truth:nd_arrat B(x) profile used to solve for ground truth

    Returns:
        dict: A dictionary containing various training results and errors.
    """
    model = initialize_model(layers,lyscl)
    x_sampled, u_sampled, h_sampled = generate_synthetic_data(N_ob, x_star, u_star, h_star, noise_level)
    equations = inverse_1st_order_equations(fractional=fractional) #set governing physics equations (1D SSA)

    training_results = train_model(model,  equations, x_sampled, u_sampled, h_sampled, 
                                   gamma, fractional,x_star,N_t,num_iterations_adam_resampled,
                                   num_iterations_adam_fixed, num_iterations_lbfgs)
    analysis_results = process_results(model, equations, x_star, u_star, h_star,
                                       x_sampled, u_sampled, h_sampled, B_truth)

    results = {**training_results, **analysis_results}
    return results

def err(B_p, B_truth):
    """
    Calculates the mean squared error between predicted and ground truth profiles.

    Args:
        B_p (np.array): Final profile predicted by the neural network.
        B_truth (np.array): Ground truth profile.

    Returns:
        float: Mean squared error.
    """
    N = B_p.size
    return (1/N) * np.sum(np.square(B_p - B_truth))

def gamma_batch(test_gammas, noise_level, x_star, 
                u_star, h_star, layers,lyscl,N_ob, fractional, N_t,
                num_iterations_adam_resampled,
                num_iterations_adam_fixed ,
                num_iterations_lbfgs, B_truth):
    """
    Tests and returns results for different values of gamma.

    Args:
        test_gammas (list): Different values of gamma to test.
        noise_level (float): Level of noise for training data.
        x_star (np.array): X-locations of training data.
        u_star (np.array): Ground truth u data.
        h_star (np.array): Ground truth h data.
        layers (list): Neural network layer configuration.
        lyscl (list): Standard deviation to set the scales for Xavier weight initialization
        N_ob (int): Number of training points.
        fractional (bool): Hyper parameters for the PINN
        N_t (int): Number of collocation points
        num_iterations_adam_resampled (int) :umber of iterations of Adam using collocation resampling
        num_iterations_adam_fixed  (int): Number of iterations of Adam with fixed collocation points
        num_iterations_lbfgs (int): Number of iterations of LBFGS using fixed collocation points  
        B_truth:nd_arrat B(x) profile used to solve for ground truth

    Returns:
        list: List of dictionaries containing results for each gamma value.
    """
    batch_results = []
    for gamma in test_gammas:
        exp_dic = Berr_func(gamma, noise_level, x_star, u_star,
                            h_star, layers,lyscl,N_ob, 
                            fractional, N_t, num_iterations_adam_resampled,
                            num_iterations_adam_fixed ,num_iterations_lbfgs,
                            B_truth)
        batch_results.append(exp_dic)
    return batch_results
