"""ACV car-localisation pipeline. Scores are relative evidence, not probabilities."""

from .data import load_case as load_case
from .features import extract_features as extract_features

__version__ = "0.1.0"


def rank_case(*args, **kwargs):
    from .inference import rank_case as implementation

    return implementation(*args, **kwargs)


def write_predictions(*args, **kwargs):
    from .inference import write_predictions as implementation

    return implementation(*args, **kwargs)
