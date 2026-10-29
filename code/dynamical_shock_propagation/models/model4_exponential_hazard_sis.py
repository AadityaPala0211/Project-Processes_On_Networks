"""Model 4: exponential-hazard SIS update."""
import numpy as np

def update(d, P, beta, recovery_rate):
    exposure = np.asarray(P.T @ d)
    hazard = -np.expm1(-beta * exposure)
    return np.clip((1.0 - recovery_rate) * d + (1.0 - d) * hazard, 0.0, 1.0)
