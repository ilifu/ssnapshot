import setuptools

with open('README.pypi.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ssnapshot',  # Replace with your own username
    version='0.0.2a1',
    author='Dane Kennedy',
    author_email='dane@idia.ac.za',
    description='A utility to generate a snapshot of slurm\'s state',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ilifu/ssnapshot',
    packages=setuptools.find_packages(),
    install_requires=[
        'coloredlogs',
        'humanize',
        'pandas',
        'jinja2',
        'tabulate',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: System Administrators',
        'Topic :: Scientific/Engineering',
        'Topic :: System :: Clustering',
        'Topic :: System :: Logging',
        'Topic :: System :: Systems Administration',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['ssnapshot=ssnapshot.command_line:main']
    },
)