


class RWrapper:
    imported_packages = {}

    def _init_r(self):
        try:
            from rpy2 import robjects
            from rpy2.robjects import numpy2ri
            from rpy2.robjects import packages
            from rpy2.robjects import vectors

            self.robjects = robjects
            self.rpackages = packages
            self.rvectors = vectors

            numpy2ri.activate()
        except ImportError:
            raise ImportError(f"Expects installed python package 'rpy2', could not find it. "
                              f"Did you install ordcomp with the r_wrapper extras? "
                              f"pip install ordcomp[r_wrapper]")


    def import_r_package(self, package_name, install_if_missing=True, **kwargs):
        if not hasattr(self, 'robjects'):
            self._init_r()

        if package_name not in RWrapper.imported_packages:
            if not self.rpackages.isinstalled(package_name):
                if install_if_missing:
                    utils = self.rpackages.importr('utils')
                    utils.chooseCRANmirror(ind=1)
                    utils.install_packages(self.rvectors.StrVector([package_name]))
                else:
                    raise ImportError(f"Expects installed R package '{package_name}', could not find it.")

            RWrapper.imported_packages[package_name] = self.rpackages.importr(package_name, **kwargs)

        setattr(self, package_name, RWrapper.imported_packages[package_name])

    def seed_r(self, random_state):
        base = self.rpackages.importr('base')
        #base.set_seed(random_state.randint(-1e9, 1e9))


