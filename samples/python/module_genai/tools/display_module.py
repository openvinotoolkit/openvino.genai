import yaml
from graphviz import Digraph
import os
import sys

def generate_pipeline_dag(yaml_data, output_filename="pipeline_dag", output_format="png"):
    """
    Parses YAML data to generate and export a DAG visualization of the pipeline.

    Args:
        yaml_data (str): The YAML string containing the pipeline configuration.
        output_filename (str): The base name for the output file.
        output_format (str): The output file format (e.g., 'png', 'pdf', 'svg').
    """
    try:
        # 1. Parse YAML Data
        config = yaml.safe_load(yaml_data)
        modules = config.get("pipeline_modules", {})
    except Exception as e:
        print(f"Failed to parse YAML data: {e}")
        return

    # 2. Initialize Graphviz Diagram
    # 
    dot = Digraph(
        name='Pipeline_DAG',
        comment='LLM-Vision Pipeline Data Flow',
        format=output_format,
        graph_attr={
            'rankdir': 'TB',  # Layout direction: TB (Top to Bottom), use 'LR' for Left to Right
            'splines': 'spline',  # Edge style: 'curved', 'spline', 'ortho'
            'bgcolor': 'white'
        },
        node_attr={
            'shape': 'box',  # Node shape: 'box', 'ellipse', etc.
            'style': 'filled',
            # Attempt to use a common font, or a fallback for cross-platform compatibility
            'fontname': 'Arial' if os.name == 'nt' else 'Helvetica', 
            'fillcolor': '#E0F7FA',  # Light blue background
            'color': '#00BCD4'  # Border color
        }
    )

    # 3. Add Nodes and Edges
    edges = []

    for module_name, module_config in modules.items():
        # Add module node with name, type, and device information
        label = f"{module_name}\n({module_config.get('type', 'Unknown')})\nDevice: {module_config.get('device', 'N/A')}"
        dot.node(module_name, label)

        # Find inputs and prepare edges
        inputs = module_config.get("inputs", [])
        for input_item in inputs:
            source_string = input_item.get("source")
            if source_string:
                # The source format is typically "source_module.output_name"
                try:
                    source_module, source_output = source_string.split('.', 1)
                except ValueError:
                    # Could not parse as 'module.output' format
                    continue

                # Check if the source module exists in the pipeline configuration
                if source_module in modules:
                    # Prepare edge: from source module to current module
                    # The edge label is the name of the data being passed
                    edge_label = source_output
                    edges.append((source_module, module_name, edge_label))

    # 4. Draw Edges with Labels
    for source, target, label in edges:
        dot.edge(source, target, label=label, color='#00BCD4', fontname='Arial' if os.name == 'nt' else 'Helvetica')


    # 5. Render and Save File
    try:
        # render() generates and saves the file
        # 'view=False' prevents automatic opening, 'cleanup=True' removes intermediate files
        dot.render(output_filename, view=False, cleanup=True)
        print(f"\n✅ Successfully generated DAG and saved as '{output_filename}.{output_format}'")
        print(f"The file is ready for viewing.")
    except Exception as e:
        # Catching the common Graphviz executable not found error
        if 'ExecutableNotFound' in str(e):
             print("\n❌ Error: Graphviz system executable (dot) not found.")
             print("Please ensure you have installed Graphviz software and added it to your system PATH.")
        else:
             print(f"\n❌ An error occurred while generating or saving the graph file: {e}")
        print(f"The graph file '{output_filename}.{output_format}' was not generated.")

if __name__ == "__main__":
    default_yaml_path = "config.yaml"
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]
    else:
        yaml_path = default_yaml_path
    print(f"yaml file path: {yaml_path}")

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_content = f.read()

        generate_pipeline_dag(yaml_content, output_format='png')
    except FileNotFoundError:
        print(f"\nError: The file '{yaml_path}' was not found. Please create it or update the path.")