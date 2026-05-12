"""infra — Infrastructure machine (device, precision, threads, memory).

Spec : DOC/CONTEXT.md §MachineProfile.

Centralise les décisions de performance liées à la machine : choix device
(cuda/cpu), precision (FP32/FP64), threads BLAS, batch cap. Source unique
de vérité — remplace les `if cuda.is_available()` éclatés dans les drivers.
"""

from infra.machine import MachineProfile, GpuArch

__all__ = ["MachineProfile", "GpuArch"]
