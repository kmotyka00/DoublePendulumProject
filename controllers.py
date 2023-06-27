import numpy as np
from control import ctrb, obsv
from numpy.linalg import matrix_rank
import scipy.linalg
class LQR:
    def __init__(self, A, B, control_time=np.inf):
        self.control_time = control_time

        self.Q = np.array([
            [1000000., .0, .0, .0, .0, .0],
            [.0, 1., .0, .0, .0, .0],
            [.0, .0, 1., .0, .0, .0],
            [.0, .0, .0, 1., .0, .0],
            [.0, .0, .0, .0, 1., .0],
            [.0, .0, .0, .0, .0, 1.],
        ])
        self.R = np.array([[2000.]])
        self.K = self.get_lqr_gains(A, B, self.Q, self.R)

    def get_control(self, state, t):
        """
        Returns control based on LQR gains and disables after time threshold
        """
        x, dx, a, da, b, db = state
        if t < self.control_time:
            _state = np.array([[x, a, b, dx, da, db]])
            return (-self.K @ _state.T)[0, 0]
        else:
            return .0
        
    def check_controllability(self, A, B):
        controllability_matrix = ctrb(A, B)
        print("Controllability: ", matrix_rank(controllability_matrix) == A.shape[0]) 
        return matrix_rank(controllability_matrix) == A.shape[0]

    def check_observability(self, A, C):
        observability_matrix = obsv(A, C)
        print("Observability: ", matrix_rank(observability_matrix) == A.shape[0])
        return matrix_rank(observability_matrix) == A.shape[0]
    
    def get_lqr_gains(self, A, B, Q, R):
        """
            Solve the continuous time lqr controller.
            dx/dt = A x + B u
            cost = integral x.T*Q*x + u.T*R*u
        """
        # Try to solve the ricatti equation
        X = np.array(scipy.linalg.solve_continuous_are(A, B, Q, R))
        # Compute the LQR gain
        K = np.array(scipy.linalg.inv(R) @ (B.T @ X))
        # eigVals, _ = scipy.linalg.eig(A - B @ K)
        return K