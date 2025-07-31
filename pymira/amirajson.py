# Converts an object loaded using AmiraMesh from https://github.com/CABI-SWS/reanimate
# There must be a folder on the python path called reanimate with the amiramesh.py file in it
import argparse
from pymira import spatialgraph
import json
from pathlib import Path
import os
join = os.path.join

def convert(filepath,opath=None,ofilename=None):
    a = spatialgraph.SpatialGraph()
    a.read(filepath,quiet=True)
    o = dict()
    # AmiraMesh object data held in fields by name:
    # 'VertexCoordinates', 'EdgeConnectivity', 'NumEdgePoints', 'EdgePointCoordinates', 'thickness'
    # Dislike the naming of thickness, so capitalize.
    # Data stored as numpy array, so call tolist()
    for field in a.fields:
        name = field['name']
        if name.lower()=='radii':
            name = 'radius'
        name = name[0].upper() + name[1:]
        if field['data'] is not None:
            o[name] = field['data'].tolist()
            
    if opath is not None:
        if ofilename is not None:
            f = join(opath,ofilename)
        else:
            f = join(opath,Path(filepath).stem+'.json')
    else:
        f = filepath.replace('.am','.json')

    with open(f, 'w') as handle:
        json.dump(o, handle, indent=4)
        
    return f

def main():
    parser = argparse.ArgumentParser(description="amirajson argument parser")

    # Add arguments
    parser.add_argument("filename", type=str, help="JSON filepath")
    parser.add_argument("-o","--opath", type=str, default=None, help="JSON filepath")
    parser.add_argument("-s","--stl", type=bool, default=True, help="STL")
    
    args = parser.parse_args()
    
    # Access the parsed arguments
    filename = args.filename
    ofile = args.opath
    stl = args.stl

    convert(filename)
    
    if stl is True:
        graph = spatialgraph.SpatialGraph()
        graph.read(filename)
        graph.export_mesh(ofile=filename.replace('.am','.stl'),resolution=10)

if __name__=='__main__':
    main()

