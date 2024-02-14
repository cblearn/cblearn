from typing import Dict, Any


class RWrapperMixin:
    imported_packages: Dict[str, Any] = {}

    @classmethod
    def init_r(cls):
        try:
            from rpy2 import robjects
            from rpy2.robjects import numpy2ri
            from rpy2.robjects import packages

            cls.robjects = robjects
            cls.rpackages = packages

            numpy2ri.activate()
        except ImportError:
            raise ImportError("Expects installed python package 'rpy2', could not find it. "
                              "Did you install cblearn with the wrapper extras? "
                              "pip install cblearn[wrapper]")

    @classmethod
    def import_r_package(cls, package_name, install_if_missing=True, **kwargs):
        if not hasattr(RWrapperMixin, 'robjects'):
            cls.init_r()

        if package_name not in RWrapperMixin.imported_packages:
            if not cls.rpackages.isinstalled(package_name):
                if install_if_missing:
                    utils = cls.rpackages.importr('utils')
                    utils.chooseCRANmirror(ind=1)
                    utils.install_packages(cls.robjects.vectors.StrVector([package_name]), verbose=False, quiet=True)
                else:
                    raise ImportError(f"Expects installed R package '{package_name}', could not find it.")

            RWrapperMixin.imported_packages[package_name] = cls.rpackages.importr(package_name, **kwargs)

        return RWrapperMixin.imported_packages[package_name]

    @classmethod
    def seed_r(cls, random_state):
        if not hasattr(RWrapperMixin, 'rpackages'):
            cls.init_r()

        base = cls.rpackages.importr('base')
        base.set_seed(random_state.randint(-1e9, 1e9))
