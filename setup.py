from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self) -> None:
        from mlsweep._parsync import fetch_parsync
        fetch_parsync()
        super().run()


setup(cmdclass={"build_py": BuildPy})
