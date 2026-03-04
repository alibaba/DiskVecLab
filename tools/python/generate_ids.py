import os
import sys
import numpy as np
import utils

def find_row_indices(subshard_data, full_data):
    # Create a dictionary for faster lookup
    full_data_dict = {}
    for i, row in enumerate(full_data):
        # Convert row to tuple so it can be used as a dictionary key
        row_tuple = tuple(row)
        if row_tuple not in full_data_dict:
            full_data_dict[row_tuple] = i
    
    # Find indices for each row in subshard_data
    indices = []
    for row in subshard_data:
        row_tuple = tuple(row)
        if row_tuple in full_data_dict:
            indices.append(full_data_dict[row_tuple])
        else:
            # This shouldn't happen if subshard is truly a subset
            raise ValueError(f"Row {row} not found in full data")
    
    return np.array(indices, dtype=np.uint32)

def process_subshard_files(directory, full_data_path, data_type):
    # Load the full data
    print(f"Loading full data from {full_data_path}")
    if data_type == 'fvecs':
        full_data = utils.read_fvecs(full_data_path)
    elif data_type == 'ivecs':
        full_data = utils.read_ivecs(full_data_path)
    elif data_type == 'bvecs':
        full_data = utils.mmap_bvecs(full_data_path)
    elif data_type in ['uint8', 'uint32', 'float']:
        if data_type == "float":
            dtype = np.float32
        elif data_type == "uint32":
            dtype = np.uint32
        elif data_type == "uint8":
            dtype = np.uint8
        full_data = utils.bin_to_numpy(dtype, full_data_path)
    else:
        raise ValueError(f"Invalid data type: {data_type}")
    
    print(f"Full data shape: {full_data.shape}")
    
    # Process each _subshard file
    for filename in os.listdir(directory):
        if filename.startswith("_subshard-") and filename.endswith(".bin") and not filename.endswith("_ids_uint32.bin"):
            subshard_path = os.path.join(directory, filename)
            print(f"Processing {subshard_path}")
            
            # Load subshard data
            if data_type == 'fvecs':
                subshard_data = utils.read_fvecs(subshard_path)
            elif data_type == 'ivecs':
                subshard_data = utils.read_ivecs(subshard_path)
            elif data_type == 'bvecs':
                subshard_data = utils.mmap_bvecs(subshard_path)
            elif data_type in ['uint8', 'uint32', 'float']:
                if data_type == "float":
                    dtype = np.float32
                elif data_type == "uint32":
                    dtype = np.uint32
                elif data_type == "uint8":
                    dtype = np.uint8
                subshard_data = utils.bin_to_numpy(dtype, subshard_path)
            
            print(f"Subshard data shape: {subshard_data.shape}")
            
            # Find row indices
            indices = find_row_indices(subshard_data, full_data)
            print(f"Found {len(indices)} indices")
            
            # Generate output filename
            leaf_id = filename[len("_subshard-"):-len(".bin")]
            output_filename = f"_subshard-{leaf_id}_ids_uint32.bin"
            output_path = os.path.join(directory, output_filename)
            
            # Save indices to bin file
            utils.numpy_to_bin(indices.reshape(-1, 1), output_path)
            print(f"Saved indices to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_ids.py <directory> <full_data_path> <data_type>")
        print("data_type can be: fvecs, ivecs, bvecs, uint8, uint32, float")
        sys.exit(1)
    
    directory = sys.argv[1]
    full_data_path = sys.argv[2]
    data_type = sys.argv[3]
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        sys.exit(1)
    
    if not os.path.exists(full_data_path):
        print(f"Full data file {full_data_path} does not exist")
        sys.exit(1)
    
    process_subshard_files(directory, full_data_path, data_type)