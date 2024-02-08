from typing import Callable, Iterable, Dict
import tensorflow as tf
from ._formulations import _data_type

class SquareLoss:
    """
    A class to calculate square loss for a physics-informed neural network (PINN)
    using fixed collocation points between iterations.

    Attributes:
        eqns (Iterable[Callable]): An iterable of callables, representing the physics-informed equations.
        eqns_data (Iterable[Callable]): An iterable of callables, representing the data equations.
        gamma (float): Normalized weighting factor for equation loss and data loss.
    """

    def __init__(self, equations: Iterable[Callable], equations_data: Iterable[Callable], gamma: float) -> None:
        self.eqns = equations
        self.eqns_data = equations_data
        self.gamma = gamma

    def __call__(self, x_eqn: tf.Tensor, data_pts: tuple, net: tf.keras.Model) -> Dict[str, tf.Tensor]:
        """
        Calculate the total loss, equation loss, and data loss.

        Args:
            x_eqn (tf.Tensor): Collocation points for the equations.
            data_pts (tuple): A tuple (x_data, y_data) for the data points.
            net (tf.keras.Model): The neural network model.

        Returns:
            Dict[str, tf.Tensor]: A dictionary containing the total loss, equation loss, and data loss.
        """
        equations = self.eqns(x=x_eqn, neural_net=net)        
        x_data, y_data = data_pts
        datas = self.eqns_data(x=x_data, y=y_data, neural_net=net)
        loss_e = sum(tf.reduce_mean(tf.square(eqn)) for eqn in equations)
        loss_d = sum(tf.reduce_mean(tf.square(data)) for data in datas)
        loss = (1 - self.gamma) * loss_d + self.gamma * loss_e
        return {"loss": loss, "loss_equation": loss_e, "loss_data": loss_d}

class SquareLossRandom:
    """
    A class to calculate square loss for a PINN while resampling the collocation points 
    after each iteration.

    Attributes:
        eqns (Iterable[Callable]): An iterable of callables, representing the physics-informed equations.
        eqns_data (Iterable[Callable]): An iterable of callables, representing the data equations.
        gamma (float): Normalized weighting factor for equation loss and data loss.
        col_gen (tf.random.Generator): A TensorFlow random number generator for collocation points.
        xmin (float): Minimum value for generating collocation points.
        xmax (float): Maximum value for generating collocation points.
        N_t (int): Number of collocation points to generate.
    """

    def __init__(self, equations: Iterable[Callable], equations_data: Iterable[Callable], gamma: float, xmin: float = 0.0, xmax: float = 1.0, N_t: int = 1001) -> None:
        self.eqns = equations
        self.eqns_data = equations_data
        self.gamma = gamma
        self.col_gen = tf.random.get_global_generator()
        self.xmin = xmin
        self.xmax = xmax
        self.N_t = N_t

    def __call__(self, x_eqn: tf.Tensor, data_pts: tuple, net: tf.keras.Model) -> Dict[str, tf.Tensor]:
        """
        Calculate the total loss, equation loss, and data loss, resampling collocation points
        at each call.    Args:
                x_eqn (tf.Tensor): Original collocation points for the equations, not used in this method.
                data_pts (tuple): A tuple (x_data, y_data) for the data points.
                net (tf.keras.Model): The neural network model.

            Returns:
                Dict[str, tf.Tensor]: A dictionary containing the total loss, equation loss, data loss, 
                                    and last collocation point (for debugging).
            """
        collocation_pts = self.xmin + (self.xmax - self.xmin) * self.col_gen.uniform(shape=[self.N_t])
        collocation_pts = tf.cast(collocation_pts**3, dtype=_data_type)
        

        x_eqn = tf.cast(collocation_pts, dtype=_data_type)
        equations = self.eqns(x=x_eqn, neural_net=net)
        x_data, y_data = data_pts
        datas = self.eqns_data(x=x_data, y=y_data, neural_net=net)
        loss_e = sum(tf.reduce_mean(tf.square(eqn)) for eqn in equations)
        loss_d = sum(tf.reduce_mean(tf.square(data)) for data in datas)
        loss = (1 - self.gamma) * loss_d + self.gamma * loss_e

        return {"loss": loss, "loss_equation": loss_e, "loss_data": loss_d, "coldebug": collocation_pts[-1]}
