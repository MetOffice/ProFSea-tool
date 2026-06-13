from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .state import ClimateState


class Component(ABC):
    @abstractmethod
    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """Calculate and return the SLR projection for this component"""
        pass


class SpatialComponent(Component):
    @property
    @abstractmethod
    def global_projection(self):
        """Spatial component must define a global_projection attribute or property"""
        pass
