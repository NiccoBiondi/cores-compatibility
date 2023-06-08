from setuptools import find_packages, setup

LICENSE = "Closed Source"
COPYRIGHT = "Copyright (c) 2023 MICC"

PACKAGES = find_packages(where="src")
entry_points = {
    'console_scripts': [
        'cores=cores.main:main',
    ],
}

setup(
    name='cores',
    version='1.0',
    description='CoReS: Compatible Representations via Stationarity',
    license=LICENSE,
    py_modules=["cores"],
    packages=PACKAGES,
    package_dir={'': 'src'},
    entry_points=entry_points,
    python_requires=">=3"
)