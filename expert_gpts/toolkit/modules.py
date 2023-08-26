import importlib
import logging
from typing import Any, Dict

from shared.config import CustomModule
from shared.patterns import Singleton

logger = logging.getLogger(__name__)


class ModuleLoader(metaclass=Singleton):
    """Module loader class"""

    _modules: Dict[str, Any] = {}

    def __init__(self):
        self._modules = {}

    def add_module(self, module: str):
        """
        Add a imported instance of the specified python package
        :param module: str:

        """
        if self.has_module(module):
            return

        self._modules[module] = importlib.import_module(module)

    def get_module(self, module: str):
        """

        :param module: str:

        """
        if not self.has_module(module):
            raise Exception(f"{module} not loaded!")

        return self._modules.get(module)

    def get_module_attr(self, module: str, attr: str):
        """

        :param module: str:
        :param attr: str:
        :param args: dict:

        """
        return getattr(self.get_module(module), attr)

    def has_module_attr(self, module: str, attr: str):
        """

        :param module: str:
        :param attr: str:

        """
        return hasattr(self.get_module(module), attr)

    def has_module(self, module: str):
        """

        :param module: str:

        """
        return module in self._modules.keys()

    def load_custom_module(self, module: CustomModule):
        return self.add_module(module.package)

    def get_custom_module(self, module: CustomModule):
        return self.get_module_attr(module.package, module.attribute)

    def build_module(self, module: CustomModule) -> CustomModule:
        self.add_module(module.package)

        executable = self.get_module_attr(module.package, module.attribute)

        if module.execute:
            executable = executable()

        module.set_attribute_built(executable)

        return module
