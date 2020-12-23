import numpy as np
from lassolver.dsolver.d_amp import amp, D_AMP

class oamp(amp):
    def __init__(self, A_p, x, snr):
        super().__init__(A_p, x, snr)


class D_OAMP(D_AMP):
    def __init__(self, A, x, snr, P):
        
