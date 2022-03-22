import setuptools

setuptools.setup(
    name="camera_toolkit",
    version="0.0.1",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    description="Package for using RBG(D) camera's in a robotics context",
    url="https://github.com/tlpss/camera-toolkit",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu",
    ],
    package_dir={"": "src"},
    install_requires = [
        "numpy",
        "opencv-contrib-python==4.5.5.62",
    ]
)