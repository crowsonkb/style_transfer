"""A module for function minimization."""

# pylint: disable=too-many-arguments, too-many-instance-attributes

import numpy as np

from num_utils import axpy, BILINEAR, dot, EPS, EWMA, resize, roll2


class AdamOptimizer:
    """Implements the Adam gradient descent optimizer [4] with iterate averaging."""
    def __init__(self, params, step_size=1, b1=0.9, b2=0.999, bp1=0, decay=0, power=1,
                 biased_g1=False):
        """Initializes the optimizer."""
        self.params = params
        self.step_size = step_size
        self.decay, self.power = decay, power

        self.i = 1
        self.xy = np.zeros(2, dtype=np.int32)
        self.g1 = EWMA.like(params, b1, correct_bias=not biased_g1)
        self.g2 = EWMA.like(params, b2)
        self.p1 = EWMA.like(params, bp1)

    def update(self, opfunc):
        """Returns a step's parameter update given a loss/gradient evaluation function."""
        # Step size decay
        step_size = self.step_size / self.i**self.power
        self.i += self.decay

        loss, grad = opfunc(self.params)

        # Adam
        self.g1.update(grad)
        self.g2.update(grad**2)
        step = self.g1.get() / (np.sqrt(self.g2.get()) + EPS)
        axpy(-step_size, step, self.params)

        # Iterate averaging
        self.p1.update(self.params)
        return roll2(self.p1.get(), -self.xy), loss

    def roll(self, xy):
        """Rolls the optimizer's internal state."""
        if (xy == 0).all():
            return
        self.xy += xy
        roll2(self.g1.value, xy)
        roll2(self.g2.value, xy)
        roll2(self.p1.value, xy)

    def set_params(self, last_iterate):
        """Sets params to the supplied array (a possibly-resized or altered last non-averaged
        iterate), resampling the optimizer's internal state if the shape has changed."""
        self.i = 1
        self.params = last_iterate
        hw = self.params.shape[-2:]
        self.g1.value = resize(self.g1.value, hw)
        self.g2.value = np.maximum(0, resize(self.g2.value, hw, method=BILINEAR))
        self.p1.value = resize(self.p1.value, hw)


class LBFGSOptimizer:
    """L-BFGS [2] for function minimization, with fixed size steps (no line search)."""
    def __init__(self, params, initial_step=0.1, n_corr=10):
        self.params = params
        self.initial_step = initial_step
        self.n_corr = n_corr
        self.xy = np.zeros(2, dtype=np.int32)
        self.loss, self.grad = None, None
        self.sk, self.yk, self.syk = [], [], []

    def update(self, opfunc):
        """Take an L-BFGS step. Returns the new parameters and the loss after the step."""
        if self.loss is None:
            self.loss, self.grad = opfunc(self.params)

        # Compute and take a step, being cautious if the L-BFGS memory is not full
        s = -self.inv_hv(self.grad)
        if not self.sk:
            s *= self.initial_step / np.mean(abs(s))
        elif len(self.sk) < self.n_corr:
            s *= len(self.sk) / self.n_corr
        self.params += s

        # Compute a curvature pair and store parameters for the next step
        loss, grad = opfunc(self.params)
        y = grad - self.grad
        self.store_curvature_pair(s, y)
        self.loss, self.grad = loss, grad

        return self.params, loss

    def store_curvature_pair(self, s, y):
        """Updates the L-BFGS memory with a new curvature pair."""
        sy = dot(s, y)
        if sy > 1e-10:
            self.sk.append(s)
            self.yk.append(y)
            self.syk.append(sy)
        if len(self.sk) > self.n_corr:
            self.sk, self.yk, self.syk = self.sk[1:], self.yk[1:], self.syk[1:]

    def inv_hv(self, p):
        """Computes the product of a vector with an approximation of the inverse Hessian."""
        p = p.copy()
        alphas = []
        for s, y, sy in zip(reversed(self.sk), reversed(self.yk), reversed(self.syk)):
            alphas.append(dot(s, p) / sy)
            axpy(-alphas[-1], y, p)

        if self.sk:
            sy, y = self.syk[-1], self.yk[-1]
            p *= sy / dot(y, y)

        for s, y, sy, alpha in zip(self.sk, self.yk, self.syk, reversed(alphas)):
            beta = dot(y, p) / sy
            axpy(alpha - beta, s, p)

        return p

    def roll(self, xy):
        """Rolls the optimizer's internal state."""
        if (xy == 0).all():
            return
        self.xy += xy
        if self.grad is not None:
            self.grad[:] = roll2(self.grad, xy)
        for s, y in zip(self.sk, self.yk):
            s[:] = roll2(s, xy)
            y[:] = roll2(y, xy)

    def set_params(self, last_iterate):
        """Sets params to the supplied array and clears the L-BFGS memory."""
        self.params = last_iterate
        self.loss, self.grad = None, None
        self.sk, self.yk, self.syk = [], [], []
