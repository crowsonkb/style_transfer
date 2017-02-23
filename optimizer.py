"""A module for function minimization."""

import numpy as np

from num_utils import axpy, BILINEAR, EPS, resize, roll2


class AdamOptimizer:
    """Implements the Adam gradient descent optimizer [4] with iterate averaging."""
    def __init__(self, params, step_size=1, b1=0.9, b2=0.999, bp1=0, decay=1, decay_power=0,
                 biased_g1=False):
        """Initializes the optimizer."""
        self.params = params
        self.step_size = step_size
        self.b1, self.b2, self.bp1 = b1, b2, bp1
        self.decay, self.decay_power = decay, decay_power
        self.biased_g1 = biased_g1

        self.i = 0
        self.step = 0
        self.xy = np.zeros(2, dtype=np.int32)
        self.g1 = np.zeros_like(params)
        self.g2 = np.zeros_like(params)
        self.p1 = np.zeros_like(params)

    def update(self, opfunc):
        """Returns a step's parameter update given a loss/gradient evaluation function."""
        # Step size decay
        step_size = self.step_size / (1 + self.decay * self.i)**self.decay_power

        self.i += 1
        self.step += 1
        loss, grad = opfunc(self.params)

        # Adam
        self.g1 *= self.b1
        axpy(1 - self.b1, grad, self.g1)
        self.g2 *= self.b2
        axpy(1 - self.b2, grad**2, self.g2)
        step_size *= np.sqrt(1 - self.b2**self.step)
        if not self.biased_g1:
            step_size /= 1 - self.b1**self.step
        step = self.g1 / (np.sqrt(self.g2) + EPS)
        axpy(-step_size, step, self.params)

        # Iterate averaging
        self.p1 *= self.bp1
        axpy(1 - self.bp1, self.params, self.p1)
        return roll2(self.p1, -self.xy) / (1 - self.bp1**self.step), loss

    def roll(self, xy):
        """Rolls the optimizer's internal state."""
        if (xy == 0).all():
            return
        self.xy += xy
        self.g1[:] = roll2(self.g1, xy)
        self.g2[:] = roll2(self.g2, xy)
        self.p1[:] = roll2(self.p1, xy)

    def set_params(self, last_iterate):
        """Sets params to the supplied array (a possibly-resized or altered last non-averaged
        iterate), resampling the optimizer's internal state if the shape has changed."""
        self.i = 0
        self.params = last_iterate
        hw = self.params.shape[-2:]
        self.g1 = resize(self.g1, hw)
        self.g2 = np.maximum(0, resize(self.g2, hw, method=BILINEAR))
        self.p1 = resize(self.p1, hw)

    def restore_state(self, optimizer):
        """Given an AdamOptimizer instance, restores internal state from it."""
        assert isinstance(optimizer, AdamOptimizer)
        self.params = optimizer.params
        self.g1 = optimizer.g1
        self.g2 = optimizer.g2
        self.p1 = optimizer.p1
        self.i = optimizer.i
        self.step = optimizer.step
        self.xy = optimizer.xy.copy()
        self.roll(-self.xy)
