import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="doduo",
    version="0.0.1",
    author="Doduo Authors",
    author_email="doduo.submission@gmail.com",
    description="Doduo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/doduo-anonymous/doduo-submission",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "transformers",
    ]
)