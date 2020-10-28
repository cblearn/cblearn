


class RWrapper:
    imported_packages = {}

    def __init__(self):
        try:
            from rpy2 import robjects
            self.robjects = robjects
            # from rpy2.robjects import numpy2ri, r
            # from rpy2.robjects.packages import importr
            # from rpy2.robjects.vectors import DataFrame
            #
            self.robjects.numpy2ri.activate()
        except ImportError:
            raise ImportError(f"Expects installed python package 'rpy2', could not find it. "
                              f"Did you install ordcomp with the r_wrapper extras? "
                              f"pip install ordcomp[r_wrapper]")

    def import_r_package(self, package_name, install_if_missing=True, **kwargs):
        if package_name not in RWrapper.imported_packages:
            if not self.robjects.packages.isinstalled(package_name):
                if install_if_missing:
                    utils = self.robjects.packages.importr('utils')
                    utils.chooseCRANmirror(ind=1)
                    utils.install_packages(self.robjects.vectors.StrVector([package_name]))
                else:
                    raise ImportError(f"Expects installed R package '{package_name}', could not find it.")

            RWrapper.imported_packages[package_name] = self.robjects.packages.importr(package_name, kwargs)

        setattr(self, package_name, RWrapper.imported_packages[package_name])


