import numpy as np

class RKSolver:
    def __init__(self, max_time, num_steps, min_time=0) -> None:
        self.times = np.linspace(min_time, max_time, num_steps)
        self.dt = self.times[1] - self.times[0]
        
    def solve(self, model, integrate_func=None):
        """
        Solves the initial-value problem of the first order ODEs
        :param initial_state: initial state
        :param times: a sequence of time points for which to solve
        :param integrate_func: calculates the next state
        :param derivative_func: computes derivatives of each state component
        :return:
        """
        if integrate_func is None:
            integrate_func = self.integrate_rk4

        states = [model.initial_state]
        for step, t in enumerate(self.times):
            states.append(integrate_func(states[-1], step, t, self.dt, model.derivatives))
        return np.array(states)


    def integrate_rk4(self, state, step, t, dt, dydx_func):
        """
        Fourth-order Runge-Kutta method.
        Source: https://www.geeksforgeeks.org/runge-kutta-4th-order-method-solve-differential-equation/
        :param step:
        :param state:
        :param t:
        :param dt:
        :param dydx_func:
        :return:
        """
        
        k1 = dydx_func(state, step, t, dt)
        k2 = dydx_func([v + d * dt / 2 for v, d in zip(state, k1)], step, t, dt)
        k3 = dydx_func([v + d * dt / 2 for v, d in zip(state, k2)], step, t, dt)
        k4 = dydx_func([v + d * dt for v, d in zip(state, k3)], step, t, dt)
        return [v + (k1_ + 2 * k2_ + 2 * k3_ + k4_) * dt / 6 for v, k1_, k2_, k3_, k4_ in zip(state, k1, k2, k3, k4)]
