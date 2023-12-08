from setuptools import Extension, setup

setup(
    name="convolution",
    ext_modules=[
        Extension(
            name="four1",
            sources=["four1.c"],
        )
    ],
)
