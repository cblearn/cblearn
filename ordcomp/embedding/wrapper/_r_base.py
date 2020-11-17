class RWrapperMixin:
    imported_packages = {}

    @classmethod
    def _init_r(cls):
        try:
            from rpy2 import robjects
            from rpy2.robjects import numpy2ri
            from rpy2.robjects import packages
            from rpy2.robjects import vectors

            cls.robjects = robjects
            cls.rpackages = packages
            cls.vectors = vectors

            numpy2ri.activate()
        except ImportError:
            raise ImportError("Expects installed python package 'rpy2', could not find it. "
                              "Did you install ordcomp with the r_wrapper extras? "
                              "pip install ordcomp[r_wrapper]")

    @classmethod
    def import_r_package(cls, package_name, install_if_missing=True, **kwargs):
        if not hasattr(RWrapperMixin, 'robjects'):
            cls._init_r()

        if package_name not in RWrapperMixin.imported_packages:
            if not cls.rpackages.isinstalled(package_name):
                if install_if_missing:
                    utils = cls.rpackages.importr('utils')
                    utils.chooseCRANmirror(ind=1)
                    utils.install_packages(cls.rvectors.StrVector([package_name]))
                else:
                    raise ImportError(f"Expects installed R package '{package_name}', could not find it.")

            RWrapperMixin.imported_packages[package_name] = cls.rpackages.importr(package_name, **kwargs)

        setattr(cls, package_name, RWrapperMixin.imported_packages[package_name])

    @classmethod
    def seed_r(cls, random_state):
        base = cls.rpackages.importr('base')
        base.set_seed(random_state.randint(-1e9, 1e9))
