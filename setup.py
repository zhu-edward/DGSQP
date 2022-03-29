from distutils.core import setup

setup(
    name="DGSQP",
    version="0.1",
    author='Edward Zhu',
    author_email='edward.zhu@berkeley.edu',
    description='A Python implementation of the Dynamic Game SQP algorithm',
    packages=['DGSQP'],
    install_requires=[
          'casadi',
          'numpy',
          'scipy',
          'matplotlib',
          'osqp',
          'pyqtgraph'
      ],
    package_dir={'DGSQP': 'DGSQP'}
)