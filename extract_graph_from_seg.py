import os
import argparse
from pathlib import Path
import numpy as np
from vessel_skeletonisation import skeletonize

def read_segmentation_tiff(path, dtype=np.uint8, validate_values=True):
    ext = os.path.splitext(path.lower())[-1]

    if ext in (".nrrd", ".nhdr", ".nii", ".mha", ".mhd"):
        # SimpleITK handles NRRD, NIfTI, and MetaImage natively.
        import SimpleITK as sitk
        sitk_img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(sitk_img)  # returns (Z, Y, X) numpy array
    elif path.lower().endswith(".nii.gz"):
        import SimpleITK as sitk
        sitk_img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(sitk_img)
    else:
        # Default: TIFF stacks
        from tifffile import imread
        arr = np.asarray(imread(path)) 
        while arr.ndim > 3 and 1 in arr.shape:
            arr = np.squeeze(arr, axis=np.argwhere(np.array(arr.shape) == 1)[0, 0])

    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D stack, got shape {arr.shape}")

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)

    if validate_values:
        uniq = np.unique(arr)
        if not set(uniq).issubset({0, 1, 2}):
            #raise ValueError(f"Unexpected labels found: {uniq}")
            print(f"Warning: Unexpected labels found in segmentation: {uniq}")

    return arr

def extract_graph_from_seg(seg_path, output_path, scale=None):

    # Load TIFF/NRDD/Nifti image
    segmentation = read_segmentation_tiff(seg_path)

    # Check binary
    if len(np.unique(segmentation)) != 2:
        raise ValueError("Segmentation must be binary.")
    if np.max(segmentation) != 1 or np.min(segmentation) != 0:
        segmentation = segmentation/np.max(segmentation) # Normalise to 0 and 1

    # Extract the graph from the segmentation
    graph = skeletonize(segmentation)

    # Create scale matrix
    if scale is not None:
        if len(scale) == 3:
            scale_matrix = np.identity(4)
            scale_matrix[0, 0] = scale[0]  # scale x-axis
            scale_matrix[1, 1] = scale[1]  # scale y-axis
            scale_matrix[2, 2] = scale[2]  # scale z-axis
            graph.scale_graph(tr=scale_matrix, radius_index=0)
        elif len(scale) == 1:
            scale_matrix = np.identity(4)
            scale_matrix[0, 0] = scale[0]  # scale x-axis
            scale_matrix[1, 1] = scale[0]  # scale y-axis
            scale_matrix[2, 2] = scale[0]  # scale z-axis
            graph.scale_graph(tr=scale_matrix, radius_index=0)
        else:
            raise ValueError("Scale must be a list of length 1 or 3.")

    # Sanity check
    check = graph.sanity_check()
    if check:
        print("Graph passed sanity check.")
    else:
        print("WARNING: Graph failed sanity check.")

    # Check treelike 
    treelike=graph.test_treelike(return_data=True)
    if treelike:
        print("Graph is treelike.")
    else:
        print("WARNING: This graph is not treelike.")
        loops = graph.identify_loops()
        num_subgraphs = graph.identify_graphs()
        print(f"Number of loops: {len(loops)}")
        print(f"Number of subgraphs: {len(num_subgraphs)}")

    # Save the graph to a file
    graph.export_mesh(ofile=output_path+".ply")


def _normalise_extensions(extensions):
    normalised = []
    for ext in extensions:
        value = ext.lower().strip()
        if not value:
            continue
        if not value.startswith("."):
            value = f".{value}"
        normalised.append(value)
    return tuple(sorted(set(normalised)))


def _matches_extensions(path, extensions):
    path_str = str(path).lower()
    return any(path_str.endswith(ext) for ext in extensions)


def _iter_input_files(input_path, recursive, extensions):
    if input_path.is_file():
        if not _matches_extensions(input_path, extensions):
            raise ValueError(f"Input file does not match supported extensions: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*" if recursive else "*"
    files = [p for p in input_path.glob(pattern) if p.is_file() and _matches_extensions(p, extensions)]
    return sorted(files)


def _file_stem(path):
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def process_path(input_path, output_dir=None, scale=None, recursive=False, extensions=None, fail_fast=False):
    # Allowed extentions for segmentation files
    extensions = (".tif", ".tiff", ".nrrd", ".nii", ".nii.gz")

    input_path = Path(input_path)
    files = _iter_input_files(input_path, recursive=recursive, extensions=extensions)

    if not files:
        raise ValueError("No valid input files found.")

    if output_dir is None:
        if input_path.is_file():
            output_dir = input_path.parent
        else:
            output_dir = os.path.join(input_path, "graphs")
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    for i, seg_file in enumerate(files):
        try:
            output_base = output_dir / _file_stem(seg_file)
            print(f"Extracting graph {i+1} of {len(files)} from segmentation file: {seg_file}")
            extract_graph_from_seg(str(seg_file), str(output_base), scale=scale)
            success += 1
        except Exception as exc:
            failed += 1
            print(f"Failed for {seg_file}: {exc}")
            if fail_fast:
                raise

    print(f"Completed. Success: {success}, Failed: {failed}")
    return success, failed


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Extract vessel skeleton graphs from segmentation files (single file or directory)."
    )
    parser.add_argument(
        "input_path",
        help="Path to a segmentation file or a directory containing segmentation files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory to store output .ply files. Defaults to input parent (file) or <input>/graphs (directory).",
    )
    parser.add_argument(
        "--scale",
        nargs="+",
        type=float,
        default=None,
        help="Voxel scale values: either one value (isotropic) or three values (x y z).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories when input_path is a directory.",
    )

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.scale is not None and len(args.scale) not in (1, 3):
        parser.error("--scale must contain either 1 value or 3 values.")

    process_path(
        input_path=args.input_path,
        output_dir=args.output_dir,
        scale=args.scale,
        recursive=args.recursive
    )


if __name__ == "__main__":
    main()