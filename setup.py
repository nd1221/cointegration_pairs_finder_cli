from setuptools import setup



setup(
    name='pairs-coint-cli',
    version='1.0',
    py_modules=[
        '__init__',
        'cli',
        'cli_helper_functions',
        'cointegration_test',
        'test_cli',
    ],
    install_requires=[
        'click',
        'datetime',
        'yfinance',
        'statsmodels',
        'pandas',
        'matplotlib',
        'scrapy',
    ],
    entry_points={'console_scripts': 'cli = cli:main'},
)