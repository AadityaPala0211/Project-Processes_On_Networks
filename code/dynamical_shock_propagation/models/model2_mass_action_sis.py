"""Model 2: bilinear / mass-action SIS update."""
import numpy as np

def update(d, P, beta, recovery_rate):
    exposure = np.asarray(P.T @ d)
    return np.clip((1.0 - recovery_rate) * d + (1.0 - d) * beta * exposure, 0.0, 1.0)
