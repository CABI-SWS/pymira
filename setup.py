from setuptools import setup

# This version keeps the flat structure of pymira and avoids import nonsense
# 1)    Bare install for just using it to convert amira files, possibly with editing (use [edit])
#       will still install the remaining modules but not dependencies.
# 2)    For complete workflows with plotting and skeletonisation, just use [full]
#       (--dry-run suggests that up to 1GB of deps will be pulled in)

# pip install 'pymira[extras] @ git+https://github.com/USERNAME/pymira.git'
# Remember the quotation marks!

extras_require = {
    'stl': ['stl'],
    'plot': ['matplotlib', 'open3d', 'pyvista', 'mayavi'],
    'image': ['scikit-image', 'nibabel', 'opencv-python'],
    'edit': ['scipy', 'dill']
}
full = list()
for reqs in extras_require.values():
    full.extend(reqs)
extras_require['full'] = full

setup(
    name='pymira',
    version='0.1',
    packages=['pymira'],
    package_dir={'pymira':''},
    install_requires=['tqdm','numpy'],
    extras_require=extras_require
)