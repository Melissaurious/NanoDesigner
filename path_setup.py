import os
import sys

def setup_all_paths():

    # Get paths from environment variables (set by your bash script)
    tool_paths = [
        os.environ.get('DIFFAB_CODE_DIR'),
        os.environ.get('DYMEAN_CODE_DIR'), 
        os.environ.get('ADESIGNER_CODE_DIR')
    ]
    
    # Add each path if it exists and isn't already in sys.path
    for path in tool_paths:
        if path and os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)

# Call it immediately when this module is imported
setup_all_paths()