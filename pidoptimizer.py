import math
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import optimizer


class PID(optimizer.Optimizer):
    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.0,
                 r: float = 1.0, kd: float = None, use_locking: bool = False,
                 name: str = "PID") -> None:
        super(PID, self).__init__(use_locking, name)

        if kd is None:
            kd = 0.25 * r + 0.5 + (1 + math.pi ** 2 * 16 / 9) / r

        self._lr = learning_rate
        self._momentum = momentum
        self._r = r 

        self._lr_t = None
        self._momentum_t = None
        self._r_t = None

        self._grad_buf = None
        self._V = None
        self._D = None

    def _create_slots(self, var_list) -> None:
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "V", self._name)
            self._zeros_slot(v, "D", self._name)
            self._zeros_slot(v, "grad_buf", self._name)

    def _prepare(self) -> None:
        self._lr_t = ops.convert_to_tensor(self._lr)
        self._momentum_t = ops.convert_to_tensor(self._momentum)
        self._r_t = ops.convert_to_tensor(self._r_t)

    def _apply_dense(self, grad, var):
        V = self.get_slot(var, 'V')
        D = self.get_slot(var, 'D')
        grad_buf = self.get_slot(var, 'grad_buf')

        V_update = V.assign(self._momentum_t * V - self._r_t * grad,
                            use_locking=self._use_locking)
        D_update = D.assign(self._momentum_t * D - (1 - self._momentum_t) *
                            (grad - grad_buf), use_locking=self._use_locking)
        grad_buf_update = grad_buf.assign(grad, use_locking=self._use_locking)

        var_update = var.assign_add(V_update + self._kd_t * D_update,
                                    use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, V_update, D_update,
                                        grad_buf_update])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

    def _finish(self, update_ops, name_scope):
        return control_flow_ops.group(update_ops, name=name_scope)
