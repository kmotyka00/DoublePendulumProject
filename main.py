import models
import solver
import simulator
import controllers
from simulation_parameters import PhysicalParameters, InitialState

InitialState.X0 = 0.0  # Cart's position
InitialState.dX0 = 0.0  # Cart's velocity
InitialState.A0 = 3  # First joint's angle
InitialState.dA0 = 0.0  # First joint's angular velocity
InitialState.B0 = 0.1  # Second joint's angle
InitialState.dB0 = 0.0  # Second joint's angular velocity

# Simple model
# model = models.SimpleModel()
# solver = solver.RKSolver(num_steps=6000, max_time=6)
# simulator = simulator.Simulator(model, solver)
# simulator.simulate()

# Inverse model
model = models.ControlledModel()
solver = solver.RKSolver(num_steps=10000, max_time=10)
simulator = simulator.Simulator(model, solver)
simulator.simulate()