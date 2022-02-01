from setuptools import find_packages, setup

setup(
    name='Alignstein',
    packages=find_packages(),
    version='0.1.0',
    description='LC/GC-MS alignment algorithm based on Wasserstein distance',
    author='Grzegorz Skoraczynski',
    license='MIT',
    python_requires=">=3.8",
    install_requires=["tqdm", "pyopenms", "numpy", "scipy", "networkx",
                      "scikit-learn"]
)
