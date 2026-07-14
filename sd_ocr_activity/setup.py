from pathlib import Path
import os

from setuptools import setup, Extension
from Cython.Build import cythonize

BASE = Path(__file__).resolve().parent
os.chdir(BASE)

extensions = [
    Extension("const", ["const.py"]),
    Extension("ocr_activity", ["ocr_activity.py"]),
    Extension("main", ["main.py"]),
    Extension("utils", ["utils.py"]),
]


setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",           
        },
        
    )
)