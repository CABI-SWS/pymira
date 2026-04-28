from pymira import spatialgraph

def inspect():
    filename = "C:\\Users\\Natal\\Documents\\CCM\\ACED\\Data\\grow\\Breast_MRI_002\\1_1_growth_15.0_graph_0_56.am"
    output_filename = "C:\\Users\\Natal\\Documents\\CCM\\ACED\\Data\\grow\\Breast_MRI_002\\1_1_growth_15.0_graph_0_56_noloops"

    # Read in a .am file
    graph = spatialgraph.SpatialGraph()
    graph.read(filename)
    graph.plot()

    # Sanity check
    check = graph.sanity_check()
    if check:
        print("Graph passed sanity check.")
    else:
        print("Graph failed sanity check.")

    # Identify loops and remove if needed
    loops = graph.identify_loops()

    if len(loops) > 0:
        print("Removing loops using default settings")
        graph.remove_loops(prefer="long")

    #Identify inlet/ outlet nodes
    a, v = graph.identify_inlet_outlet()
    print(f"Inlet node: {a}")
    print(f"Outlet node: {v}")

    # Test for treelike branching
    treelike=graph.test_treelike(return_data=True)
    if treelike:
        print("This graph is treelike.")
    else:
        print("This graph is not treelike. Removing subgraphs and retesting.")
        #graph.remove_subsidiary_graphs()
        treelike=graph.test_treelike()
        if treelike:   
            print("This graph is now treelike.")
        else:
            print("This graph is still not treelike. Continuing with non-treelike graph.")

    # Save the modified graph
    graph.write(output_filename+".am")
    graph.export_mesh(ofile=output_filename+".ply")
    print(f"Modified graph saved to {output_filename}")

    # Display the edited graph
    graph.plot()

if __name__=='__main__':
    inspect()