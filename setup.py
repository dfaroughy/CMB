
import setuptools

setuptools.setup(name="cmb",
                version=1.0,
                url="git@github.com:dfaroughy/cmb.git",
                packages=setuptools.find_packages("src"),
                package_dir={"": "src"}
                )