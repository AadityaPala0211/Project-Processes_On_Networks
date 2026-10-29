"""Four comparable discrete-time disruption models."""
from .model1_clipped_linear import update as model1_update
from .model2_mass_action_sis import update as model2_update
from .model3_mmca_sis import update as model3_update
from .model4_exponential_hazard_sis import update as model4_update

UPDATES = {
    "model1_linear": model1_update,
    "model2_bilinear_sis": model2_update,
    "model3_mmca_sis": model3_update,
    "model4_exponential_sis": model4_update,
}
