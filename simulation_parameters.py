class PhysicalParameters:
    M = 5.0
    m1 = 2.0
    m2 = 1.5
    l1 = 0.5
    l2 = 0.25
    g = 9.81

class InitialState:
    X0 = 0.0   # Cart's position
    dX0 = 0.0  # Cart's velocity
    A0 = -0.2  # First joint's angle
    dA0 = 0.0  # First joint's angular velocity
    B0 = 0.1   # Second joint's angle
    dB0 = 0.0  # Second joint's angular velocity