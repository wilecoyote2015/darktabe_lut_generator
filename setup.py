import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="darktable-lut-generator",
    version="0.0.12",
    author="Björn Sonnenschein",
    author_email="wilecoyote2015@gmail.com",
    description="Estimate a .cube 3D lookup table from camera images for the Darktable lut 3D module.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wilecoyote2015/darktabe_lut_generator",
    project_urls={
        "Bug Tracker": "https://github.com/wilecoyote2015/darktabe_lut_generator/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'darktable_lut_generator=darktable_lut_generator.main:main',
            'darktable_lut_generate_pattern=darktable_lut_generator.make_rgb_image:main'
        ]
    },
    install_requires=[
        'numpy',
        'sklearn',
        'opencv-python',
        'tqdm',
        'plotly',
        'pandas',
    ]
)
