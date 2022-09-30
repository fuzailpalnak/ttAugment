from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
install_requires = [
    "imgaug == 0.4.0",
    "numpy == 1.19.1",
    "image-fragment==0.2.2",
    "py-oneliner == 0.0.1",
]

setup(
    name="ttAugment",
    version="0.4.1",
    author="Fuzail Palnak",
    author_email="fuzailpalnak@gmail.com",
    description="Test Time Augmentations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires="~=3.6",
    install_requires=install_requires,
    keywords=["Deep Learning Inference", "PyTorch", "Augmentations", "TensorFlow"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
