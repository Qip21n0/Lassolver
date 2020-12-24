import numpy as np
from lassolver.utils.func import *
from lassolver.dsolver.d_amp import amp, D_AMP

class oamp(amp):
    def __init__(self, A, x, snr, W):
        super().__init__(A_p, x, snr)
        self.W = W

    def local_compute(self):
        pass


class D_OAMP(D_AMP):
    def __init__(self, A, x, snr, P):
        pass
