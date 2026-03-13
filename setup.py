from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self) -> None:
        super().run()
        import sys
        sys.path.insert(0, self.build_lib)
        from mlsweep._parsync import fetch_parsync
        fetch_parsync()


setup(cmdclass={"build_py": BuildPy})
