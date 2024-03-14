import tensorflow as tf
import numpy as np
from pyDOE import lhs  # For Latin Hypercube Sampling

_data_type = tf.float64


def data_equations(x, y, neural_net):
    """
    Calculate the difference between neural network predictions and observed data for training PINNs.

    Args:
        x (tf.Tensor): The input tensor for the neural network.
        y (tuple): A tuple of observed data (u_data, h_data).
        neural_net (tf.keras.Model): The neural network model for prediction.

    Returns:
        tuple: Differences between observed data and neural network predictions.
    """
    u_data, h_data = y
    nn_forward = neural_net(x)
    u_pred = nn_forward[..., 0]
    h_pred = nn_forward[..., 1]
    return u_data - u_pred, h_data - h_pred

def inverse_1st_order_equations( fractional:bool,   spy = 60 * 60 * 24 * 365.25, 
                                rhoi=910,rhow=1028,g=9.81,H0=1.0e3,B0=1.4688e8,n=3):
 

    """
    Create a function representing inverse first-order equations for use in a PINN.

    Args:
        fractional (bool): If True, use the fractional form of the equation.
        spy = 60 * 60 * 24 * 365.25
        rhoi = 910.
        rhow = 1028.
        g = 9.81
        H0 = 1.0e3
        B0 = 1.4688e8
        n = 3

    Returns:
        function: A function that calculates the inverse first-order equations.
    """
    delta = 1. - rhoi / rhow
    g = 9.81
    a = 0.3 / spy
    Q0 = 4.0e5 / spy
    Z0 = a ** (1/(n+1)) * (4 * B0) ** (n / (n + 1)) / (rhoi * g * delta) ** (n/(n + 1))
    U0 = 400 / spy
    Lx = U0 * Z0 / a
    h0 = H0 / Z0; q0 = Q0 / (U0 * Z0)
    nu_star = (2 * B0) / ( rhoi * g * delta * Z0) * (U0 / Lx) ** (1 / n)
    A0 = (a * Lx) / (U0 * Z0)
    
    def inverse_1st_order(x, neural_net=None, drop_mass_balance: bool = True):
        """
        The inverse first-order equation calculation.

        Args:
            x (tf.Tensor): The input tensor.
            neural_net (tf.keras.Model): The neural network model for prediction.
            drop_mass_balance (bool): Whether to include mass balance equation in the calculation.

        Returns:
            tuple: Calculated values from the inverse first-order equations.
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            nn_forward = neural_net(x)
            u = nn_forward[..., 0:1]
            h = nn_forward[..., 1:2]
            B = nn_forward[..., 2:3]
            uh = h * u

        u_x = tape.gradient(u, x)
        uh_x = tape.gradient(uh, x)

        # Momentum balance governing equation
        momentum_balance = (2 * nu_star * B) ** n * u_x - h ** n if not fractional \
                           else 2 * nu_star * B * (tf.abs(u_x) ** (1/n - 1)) * u_x - h

        if drop_mass_balance:
            return (momentum_balance,)
        else:
            mass_balance = uh_x - A0
            return momentum_balance, mass_balance

    return inverse_1st_order

def to_mat_tensor(var):
    """
    Converts a variable to a TensorFlow constant with a specific shape and data type for MATLAB compatibility.
    Args:
        var: Variable to be converted.

    Returns:
        tf.Tensor: A TensorFlow tensor of the variable.
    """
    return tf.constant([var], dtype=_data_type)[None, :]


def to_tensor(var):
    """
    Converts a variable to a TensorFlow constant with the specified data type.
    Args:
        var: Variable to be converted.

    Returns:
        tf.Tensor: A TensorFlow tensor of the variable.
    """
    return tf.constant(var, dtype=_data_type)

def analytic_h_constantB(x):
    """
    Computes the analytic solution for h with a constant B value.
    Args:
        x (array-like): The x values for which the solution is computed.

    Returns:
        np.array: The computed values of h.
    """
    return ((A0 * h0 ** (n + 1) * (A0 * x + q0) ** (n + 1)) / 
            (A0 * q0 ** (n + 1) - (q0 * h0) ** (n + 1) + (h0 * (A0 * x + q0)) ** (n + 1))) ** (1 / (n + 1))

def analytic_u_constantB(x):
    """
    Computes the analytic solution for u with a constant B value.
    Args:
        x (array-like): The x values for which the solution is computed.

    Returns:
        np.array: The computed values of u.
    """
    return (A0 * x + q0) / analytic_h_constantB(x)

def get_collocation_points(x_train, xmin: float, xmax: float, N_t: int):
    """
    Generates collocation points for training using Latin Hypercube Sampling.
    Args:
        x_train (tf.Tensor): Tensor of training points.
        xmin (float): Minimum value of the range for generating points.
        xmax (float): Maximum value of the range for generating points.
        N_t (int): Number of collocation points to generate.

    Returns:
        tf.Tensor: Tensor of generated collocation points.
    """
    collocation_pts = xmin + (xmax - xmin) * lhs(1, N_t) ** 3
    return tf.cast(collocation_pts, dtype=_data_type)
