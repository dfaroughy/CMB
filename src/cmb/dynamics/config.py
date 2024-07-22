from dataclasses import dataclass

""" Default configurations for continious normalizing flow dynamics.
"""

@dataclass
class CondFlowMatch_Config:
    DYNAMICS : str = 'CFM'
    SIGMA: float = 1e-5
    AUGMENTED : bool = False
    T0 : float = 0.0
    T1 : float = 1.0
