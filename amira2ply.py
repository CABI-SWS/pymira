# Converts an object loaded using AmiraMesh from https://github.com/CABI-SWS/reanimate
# There must be a folder on the python path called reanimate with the amiramesh.py file in it
import argparse
from pymira import spatialgraph
import json
from pathlib import Path
import os
join = os.path.join

def convert(filepath,ofilename=None):
    a = spatialgraph.SpatialGraph()
    a.read(filepath,quiet=True)
    a.export_mesh(ofilename)

def main():
    parser = argparse.ArgumentParser(description="amirajson argument parser")

    # Add arguments
    parser.add_argument("filename", type=str, help="JSON filepath")
    parser.add_argument("-o","--ofile", type=str, default=None, help="Mesh filepath")
    
    args = parser.parse_args()
    
    # Access the parsed arguments
    filename = args.filename
    ofile = args.ofile

    convert(filename,ofilename=ofile)

if __name__=='__main__':
    main()

