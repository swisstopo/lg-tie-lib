import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="untie",
    version=read("VERSION").strip(),
    author="Anna Rauch",
    description=("Trace Information Extraction (TIE) library"),
    author_email="contact@infogrip.ch",
    maintainer="Marc Monnerat",
    maintainer_email="marc.monnerat@swisstopo.ch",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    packages=["untie", "untie/scripts"],
    license_files=("LICENSE.md",),
    install_requires=[
        "shapely>=2.0.0",
        "geocube",
        "numpy",
        "geopandas",
        "mayavi",
        "matplotlib",
        "rasterio",
        "scipy",
        "scikit-image",
    ],
    package_data={
        "untie": [
            "src",
            'src/untie/data/swissALTI3d/*',
        ],
        "untie/scripts": [ 'scripts/*',],
       
    },
      
   include_package_data=True,
    entry_points={
        'console_scripts': [
            'untie_demo = untie.scripts.untie_demo:main',
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: GIS",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
