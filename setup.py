"""
Simple setup.py file so that the `information_retrieval subdirectory can be installed as a package.
This way, relative imports can be used within the package, which is nice because I modularized the
code base. So make sure to pip3 install this in the virtual environment you made.
"""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(name="lxmert_clip_tiir", version="1.0", packages=find_packages())
