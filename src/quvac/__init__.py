# ruff: noqa: F401
import quvac.analytic_scalings
from quvac.cluster.gridscan import main_gridscan
import quvac.config
import quvac.field
import quvac.grid
import quvac.integrator
import quvac.log
import quvac.postprocess
from quvac.simulation import main_simulation
from quvac.simulation_parallel import main_simulation_parallel
import quvac.utils
from quvac.utils import find_classes_in_package

try:
    from quvac.cluster.optimization import main_optimization
except ModuleNotFoundError:
    def main_optimization():
        """
        Placeholder function for optimization main function.
        """
        print("Optimization module not available. Skipping optimization.")

__cls_names__ = None
if __cls_names__ is None:
    __cls_names__ = find_classes_in_package("quvac")

__doc_const_in_modules__ = [
    "config",
    "field",
]