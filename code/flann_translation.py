from distutils.sysconfig import get_python_lib
import os


if __name__ == '__main__':
    path = get_python_lib()
    os.system(f'2to3 -w {path}/pyflann')
