import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EssayGrader", # Replace with your own username
    version="0.0.1",
    author="Sunguk Keem",
    author_email="keem@keem.net",
    description="AI Engine Grading Essay",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keemsunguk/EGrader/tree/master/egrader",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={'': ['data/dictionary.txt']},
)