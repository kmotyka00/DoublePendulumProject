import numpy as np
from abc import ABC, abstractmethod
from controllers import LQR
from simulation_parameters import PhysicalParameters, InitialState


class Model(ABC):
    def __init__(self, physical_parameters=PhysicalParameters(), initial_state=InitialState(), cart_frozen=False) -> None:

        # Physical parameters
        self.M = physical_parameters.M
        self.m1 = physical_parameters.m1
        self.m2 = physical_parameters.m2
        self.l1 = physical_parameters.l1
        self.l2 = physical_parameters.l2
        self.g = physical_parameters.g

        # Initial state
        self.X0 = initial_state.X0    # Cart's position
        self.dX0 = initial_state.dX0  # Cart's velocity
        self.A0 = initial_state.A0    # First joint's angle
        self.dA0 = initial_state.dA0  # First joint's angular velocity
        self.B0 = initial_state.B0    # Second joint's angle
        self.dB0 = initial_state.dB0  # Second joint's angular velocity

        self.initial_state = np.array([self.X0, 
                                       self.dX0,
                                       self.A0,
                                       self.dA0,
                                       self.B0,
                                       self.dB0])
        
        self.cart_frozen = cart_frozen

    @abstractmethod
    def derivatives(self, state, step, t_, dt_):
        pass

class SimpleModel(Model):

    def derivatives(self, state, step, t_, dt_):
        x, dx, a, da, b, db = state


        dL_dx = 0.0
        dL_da = -(self.m1 + self.m2) * self.l1 * da * dx * np.sin(a) + (self.m1 + self.m2) * self.g * self.l1 * np.sin(a) - self.m2 * self.l1 * self.l2 * da * db * np.sin(a - b)
        dL_db = self.m2 * self.l2 * (self.g * np.sin(b) + self.l1 * da * db * np.sin(a - b) - dx * db * np.sin(b))

        A = np.zeros((3, 3))
        b_vec = np.zeros(3)

        A[0, 0] = self.M + self.m1 + self.m2
        A[0, 1] = (self.m1 + self.m2) * self.l1 * np.cos(a)
        A[0, 2] = self.m2 * self.l2 * np.cos(b)

        A[1, 0] = (self.m1 + self.m2) * self.l1 * np.cos(a)
        A[1, 1] = (self.m1 + self.m2) * self.l1 ** 2
        A[1, 2] = self.m2 * self.l1 * self.l2 * np.cos(a - b)

        A[2, 0] = self.m2 * self.l2 * np.cos(b)
        A[2, 1] = self.m2 * self.l1 * self.l2 * np.cos(a - b)
        A[2, 2] = self.m2 * self.l2 ** 2

        b_vec[0] = (self.m1 + self.m2) * self.l1 * da ** 2 * np.sin(a) + self.m2 * self.l2 * db ** 2 * np.sin(b) + dL_dx
        b_vec[1] = (self.m1 + self.m2) * dx * da * self.l1 * np.sin(a) + self.m2 * self.l1 * self.l2 * db * (da - db) * np.sin(a - b) + dL_da
        b_vec[2] = self.m2 * dx * db * self.l2 * np.sin(b) + self.m2 * self.l1 * self.l2 * da * (da - db) * np.sin(a - b) + dL_db

        ddx, dda, ddb = np.linalg.solve(A, b_vec)

        if self.cart_frozen:
            dx = 0.0
            ddx = 0.0

        return np.array([dx, ddx, da, dda, db, ddb])

class ControlledModel(Model):

    def __init__(self, controller=None, cart_frozen=False) -> None:
        super().__init__(cart_frozen=cart_frozen)

        # Controller
        if controller is None:
            A, B = self.get_space_state_representation()    
            self.controller = LQR(A, B)
        else:
            self.controller = controller

    def derivatives(self, state, step, t_, dt_):
        x, dx, a, da, b, db = state

        u = self.controller.get_control(state, t_)
        
        dL_dx = 0.0
        dL_da = -(self.m1 + self.m2) * self.l1 * da * dx * np.sin(a) + (self.m1 + self.m2) * self.g * self.l1 * np.sin(a) - self.m2 * self.l1 * self.l2 * da * db * np.sin(a - b)
        dL_db = self.m2 * self.l2 * (self.g * np.sin(b) + self.l1 * da * db * np.sin(a - b) - dx * db * np.sin(b))

        A = np.zeros((3, 3))
        b_vec = np.zeros(3)

        A[0, 0] = self.M + self.m1 + self.m2
        A[0, 1] = (self.m1 + self.m2) * self.l1 * np.cos(a)
        A[0, 2] = self.m2 * self.l2 * np.cos(b)

        A[1, 0] = (self.m1 + self.m2) * self.l1 * np.cos(a)
        A[1, 1] = (self.m1 + self.m2) * self.l1 ** 2
        A[1, 2] = self.m2 * self.l1 * self.l2 * np.cos(a - b)

        A[2, 0] = self.m2 * self.l2 * np.cos(b)
        A[2, 1] = self.m2 * self.l1 * self.l2 * np.cos(a - b)
        A[2, 2] = self.m2 * self.l2 ** 2

        b_vec[0] = (self.m1 + self.m2) * self.l1 * da ** 2 * np.sin(a) + self.m2 * self.l2 * db ** 2 * np.sin(b) + dL_dx + u
        b_vec[1] = (self.m1 + self.m2) * dx * da * self.l1 * np.sin(a) + self.m2 * self.l1 * self.l2 * db * (da - db) * np.sin(a - b) + dL_da
        b_vec[2] = self.m2 * dx * db * self.l2 * np.sin(b) + self.m2 * self.l1 * self.l2 * da * (da - db) * np.sin(a - b) + dL_db

        ddx, dda, ddb = np.linalg.solve(A, b_vec)


        return np.array([dx, ddx, da, dda, db, ddb])
    
    def get_space_state_representation(self):

        X = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, self.M + self.m1 + self.m2, self.l1 * (self.m1 + self.m2), self.l2 * self.m2],
            [0, 0, 0, self.l1 * (self.m1 + self.m2), self.l1 ** 2 * (self.m1 + self.m2), self.l1 * self.l2 * self.m2],
            [0, 0, 0, self.l2 * self.m2, self.l1 * self.l2 * self.m2, self.l2 ** 2 * self.m2]
        ], dtype="float64")
        
        N = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, (self.m1 + self.m2) * self.l1 * self.g, 0, 0, 0, 0],
            [0, 0, self.m2 * self.l2 * self.g, 0, 0, 0]
        ], dtype="float64")

        F = np.array([[0, 0, 0, 1, 0, 0]], dtype="float64").T
        X_inv = np.linalg.inv(X)
        A = X_inv @ N
        B = np.linalg.inv(X) @ F

        return A, B


