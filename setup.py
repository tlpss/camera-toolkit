import setuptools

setuptools.setup(
    name="camera_toolkit",
    version="0.0.1",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    description="Package for using RGB(D) camera's in a robotics context",
    url="https://github.com/tlpss/camera-toolkit",
    packages=["camera_toolkit"],
    install_requires=[
        "numpy",
        "opencv-contrib-python==4.5.5.62",
    ],
)
