#!/usr/bin/env python3
"""
Run training dataset creation with updated geojson loading and early validation.
"""

import os
import sys
import argparse

# Add current directory to path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training_dataset_composer import (
    create_training_data,
    normalize_graph_path,
    ENHANCED_SUBGRAPH_DIR_NAME,
)
import json

# Get base paths from environment variables
DATA_ROOT = os.getenv('URBANKG_DATA_ROOT', './data')
OUTPUT_ROOT = os.getenv('URBANKG_OUTPUT_ROOT', './output')

def detect_resume_point(jsonl_file_path: str) -> int:
    """
    Auto-detect resume point by checking which subgraphs have already been enhanced.
    
    Args:
        jsonl_file_path: Path to the JSONL file
        
    Returns:
        Line number to resume from (0-based), or 0 if starting from beginning
    """
    print("🔍 Auto-detecting resume point...")
    
    if not os.path.exists(jsonl_file_path):
        print(f"❌ JSONL file not found: {jsonl_file_path}")
        return 0
    
    enhanced_subgraphs = set()
    subgraph_base_dir = None
    
    # First, find all enhanced subgraphs
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                first_data = json.loads(first_line)
                graph_path = first_data.get('graph_path', '')
                if graph_path:
                    # Normalize the graph path to local directory structure
                    normalized_graph_path = normalize_graph_path(graph_path)
                    subgraph_base_dir = os.path.dirname(normalized_graph_path)
                    # subgraph_base_dir= subgraph_base_dir.replace('/subgraph_data','')
                    enhanced_dir = os.path.join(subgraph_base_dir, ENHANCED_SUBGRAPH_DIR_NAME)
                    if os.path.exists(enhanced_dir):
                        enhanced_files = os.listdir(enhanced_dir)
                        enhanced_subgraphs.update({f for f in enhanced_files if f.endswith('.pkl')})
                        print(
                            f"Found {len(enhanced_subgraphs)} already enhanced subgraphs "
                            f"in {enhanced_dir}"
                        )
                    else:
                        print("No enhanced directory found - starting from beginning")
                        return 0
    except Exception as e:
        print(f"Error detecting enhanced subgraphs: {e}")
        return 0
    
    if not enhanced_subgraphs:
        print("No enhanced subgraphs found - starting from beginning")
        return 0
    
    # Now find the last processed line
    last_processed_line = 0
    processed_count = 0
    
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    graph_path = data.get('graph_path', '')
                    
                    if graph_path:
                        # Normalize the graph path to local directory structure
                        normalized_graph_path = normalize_graph_path(graph_path)
                        pkl_filename = os.path.basename(normalized_graph_path)
                        if pkl_filename in enhanced_subgraphs:
                            last_processed_line = line_num
                            processed_count += 1
                        else:
                            # Found first unprocessed item
                            break
                            
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        print(f"Error scanning JSONL file: {e}")
        return 0
    
    print(f"📊 Found {processed_count} processed entries")
    print(f"🎯 Resume from line {last_processed_line + 2} (after last processed)")
    
    return last_processed_line + 1  # Resume after the last processed line

def main():
    """Run training dataset creation for both JSONL files."""
    parser = argparse.ArgumentParser(description="Run training dataset creation with auto-resume")
    parser.add_argument("--resume-from", type=int, default=None, 
                       help="Manually specify resume line (0-based). If not provided, auto-detects.")
    parser.add_argument("--no-resume", action="store_true", 
                       help="Start from beginning, ignoring existing enhanced subgraphs")
    parser.add_argument("--file", type=str, choices=["qa3", "qa4", "both"], default="both",
                       help="Which file to process: qa3, qa4, or both")
    parser.add_argument("--data-folder", type=str, default="newyork",
                       help="Data folder name (e.g., 'singapore', 'newyork', 'beijing')")
    parser.add_argument("--no-enhance", action="store_true",
                       help="Disable graph extension (still creates node_text attributes, but doesn't extend graph size)")
    
    args = parser.parse_args()
    
    print("🚀 Starting Training Dataset Creation with Auto-Resume")
    print("=" * 80)
    
    if args.no_resume:
        print("🔄 No-resume mode: Starting from beginning")
    elif args.resume_from is not None:
        print(f"🔄 Manual resume mode: Starting from line {args.resume_from + 1}")
    else:
        print("🔄 Auto-resume mode: Will detect resume point automatically")
    
    if args.no_enhance:
        print("⚡ No-enhance mode: Creating node_text attributes without extending graph size")
    else:
        print("⚡ Graph extension enabled: Will extend graphs with Mapillary nodes and create node_text attributes")
    
    # Determine data folder and base path
    data_folder = args.data_folder
    base_data_path = os.path.join(DATA_ROOT, 'geo', 'SR', 'osm_data', data_folder)
    
    print(f"📂 Data folder: {data_folder}")
    print(f"📂 Base data path: {base_data_path}")
    
    all_files = [
        # {
        #         "name": "qa4",
        #         "input": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes0_reversed_800m_new.jsonl",
        #         "output": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes0_reversed_800m_new_{data_folder}.jsonl"
        #     },
        # {
        #     "name": "qa4",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes0_reversed_2km_new1.jsonl",
        #     "output": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes0_reversed_2km_new1_{data_folder}.jsonl"
        # },
        # {
        #     "name": "qa4",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes0_reversed_2km_new.jsonl",
        #     "output": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes0_reversed_2km_new_{data_folder}.jsonl"
        # },
        #
        #
        # {
        #     "name": "qa4",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_qa4_with_intersetion.jsonl",
        #     "output": f"{base_data_path}/training_data_qa4_with_intersection_{data_folder}.jsonl"
        # },
        # {
        #     "name": "qa43",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_qa3_all_point.jsonl",
        #     "output": f"{base_data_path}/reasoning_path_mapillary_swift_qa3_all_point_{data_folder}.jsonl"
        # },
        # {
        #     "name": "qa43",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_qa3_no_intersection_nodes_1.jsonl",
        #     "output": f"{base_data_path}/reasoning_path_mapillary_swift_qa3_no_intersection_nodes_1_{data_folder}.jsonl"
        # },
        # {
        #     "name": "qa3",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes0_reversed.jsonl",
        #     "output": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes0_reversed_{data_folder}.jsonl"
        # },
        # {
        #     "name": "qa3",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes1_reversed.jsonl",
        #     "output": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes1_reversed_{data_folder}.jsonl"
        # },
        # #
        # {
        #     "name": "qa3",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes2_reversed.jsonl",
        #     "output": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes2_reversed_{data_folder}.jsonl"
        # },
        # {
        #     "name": "qa3",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes0_reversed_2km_new2.jsonl",
        #     "output": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes0_reversed_2km_new2_{data_folder}.jsonl"
        # },
        # {
        #     "name": "qa3",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes1_reversed_700m.jsonl",
        #     "output": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes1_reversed_700m_{data_folder}.jsonl"
        # },
        # {
        #     "name": "qa3",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes1_reversed_1km_last.jsonl",
        #     "output": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes1_reversed_1km_last_{data_folder}.jsonl"
        # },
        {
            "name": "qa3",
            "input":f"{base_data_path}/reasoning_path_mapillary_visualization.jsonl",
            "output":f"{base_data_path}/reasoning_path_mapillary_visualization_{data_folder}.jsonl",
        },

        # {
        #     "name": "qa3",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes_reversed_2km_new.jsonl",
        #     "output": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes_reversed_2km_new_{data_folder}.jsonl"
        # },
        # {
        #     "name": "qa3",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes_reversed_800m.jsonl",
        #     "output": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes_reversed_800m_{data_folder}.jsonl"
        # },
        # {
        #     "name": "qa3",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes1_reversed_1km_last.jsonl",
        #     "output": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes1_reversed_1km_last_{data_folder}.jsonl"
        # },
        # {
        #     "name": "qa3",
        #     "input": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes_reversed_2km_for_SR.jsonl",
        #     "output": f"{base_data_path}/reasoning_path_mapillary_swift_no_intersection_nodes_reversed_2km_for_SR_{data_folder}.jsonl"
        # },


    ]
    
    # Filter files based on command-line argument
    # if args.file == "qa3":
    #     files_to_process = [f for f in all_files if f["name"] == "qa3"]
    # elif args.file == "qa4":
    #     files_to_process = [f for f in all_files if f["name"] == "qa4"]
    # else:  # both
    files_to_process = all_files
    
    total_processed = 0
    
    for i, file_config in enumerate(files_to_process, 1):
        input_file = file_config["input"]
        output_file = file_config["output"]
        
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            continue
            
        print(f"\n📁 Processing File {i}/2: {os.path.basename(input_file)}")
        print("-" * 80)
        
        # try:
        # Determine resume point based on command-line arguments
        if args.no_resume:
            resume_line = 0
            print("🔄 Starting from beginning (no-resume mode)")
        elif args.resume_from is not None:
            resume_line = args.resume_from
            print(f"🔄 Manual resume from line {resume_line + 1}")
        else:
            # Auto-detect resume point based on existing enhanced subgraphs
            resume_line = detect_resume_point(input_file)
            if resume_line > 0:
                print(f"🔄 Auto-detected resume point: line {resume_line + 1}")
            else:
                print("🔄 Starting from beginning (no previous progress found)")

        training_count = create_training_data(
            jsonl_file_path=input_file,
            output_file_path=output_file,
            enhance_subgraphs=True,  # Always enhance to create node_text attributes
            compass_file_path=None,  # Auto-constructed from data_folder
            data_folder=data_folder,  # Pass data folder to function
            resume_from_line=resume_line,
            extend_graph=not args.no_enhance  # Extend graph unless --no-enhance is set
        )

        total_processed += training_count
        print(f"✅ Successfully processed {training_count} training examples")
        print(f"📄 Output saved to: {output_file}")
            
        # except Exceptionon as e:
        #     print(f"❌ Error processing {input_file}: {e}")
        #     import traceback
        #     traceback.print_exc()
    
    print(f"\n" + "=" * 80)
    print(f"🎉 TRAINING DATASET CREATION COMPLETED")
    print(f"=" * 80)
    print(f"Total training examples created: {total_processed:,}")
    print(f"Files processed: {len(files_to_process)}")
    
    # Show output files
    print(f"\n📋 Output Files:")
    for file_config in files_to_process:
        output_file = file_config["output"]
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"   ✅ {output_file} ({file_size:.1f} MB)")
        else:
            print(f"   ❌ {output_file} (not created)")

if __name__ == "__main__":
    # Print usage examples if no arguments provided
    if len(sys.argv) == 1:
        print("🚀 Training Dataset Creation with Auto-Resume")
        print("=" * 60)
        print("Usage examples:")
        print("  python run_training_creation.py                              # Auto-detect resume point (Singapore)")
        print("  python run_training_creation.py --data-folder newyork        # Process New York data")
        print("  python run_training_creation.py --data-folder beijing        # Process Beijing data")
        print("  python run_training_creation.py --no-resume                  # Start from beginning")
        print("  python run_training_creation.py --resume-from 1500           # Resume from line 1501")
        print("  python run_training_creation.py --no-enhance                 # Run without extending graphs")
        print("  python run_training_creation.py --file qa4                   # Process only qa4 file")
        print("  python run_training_creation.py --file qa3                   # Process only qa3 file")
        print("  python run_training_creation.py --data-folder newyork --file qa3  # New York qa3 only")
        print("  python run_training_creation.py --data-folder newyork --no-enhance  # New York without enhancement")
        print("  python run_training_creation.py --help                       # Show full help")
        print()
    
    main()
