# pymira

Python library for spatial graph analysis

## Overview

**pymira** is a Python package for reading, analysing, and visualising spatial graphs, particularly those representing vascular and neural structures in 3D. It is designed for scientific research, allowing you to:

* Load, manipulate, and analyse spatial graphs from various sources
* Compute structural metrics and statistics
* Perform graph-based analyses such as connectivity, pathfinding, clustering, and skeletonisation
* Visualise graphs in 2D/3D

## Features

* **Spatial graph data model**: Load and save spatial graphs from custom or AmiraMesh formats
* **Graph analysis**: Degree, path length, clustering, connectivity, skeletonisation, and more
* **Geometry operations**: Compute distances, midlines, and geometric transformations
* **Visualisation**: Plot graphs and overlays for publication-quality figures

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/pymira.git
cd pymira
pip install .
```

## Quickstart

```python
from pymira import spatialgraph

# Load a spatial graph
g = spatialgraph.SpatialGraph()
g.read('my_graph_file.am')

# Analyse graph properties
print(g.num_nodes(), g.num_edges())
metrics = g.compute_statistics()

# Visualise
g.plot()
```

## Applications

* Vascular and neural network analysis in biological imaging
* Structural comparison between 3D networks
* Automated quantification and feature extraction

## Contributing

Contributions are welcome! Please submit issues or pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

\[MIT Lice

