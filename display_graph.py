import sys
from pymira import spatialgraph

print("sys.argv =", sys.argv)

if len(sys.argv) > 1:
    filename = ' '.join(sys.argv[1:])
    print(f"You opened: {filename}")
    # Add your file-handling logic here
    graph = spatialgraph.SpatialGraph()
    graph.read(filename)
    graph.plot()