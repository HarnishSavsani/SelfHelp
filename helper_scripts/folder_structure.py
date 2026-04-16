import os
from pathlib import Path

def generate_directory_tree(root_dir, indent=""):
    """
    Recursively generates a directory tree string.
    
    Args:
        root_dir (Path): The directory to crawl.
        indent (str): Current indentation level.
    """
    # Folders to completely ignore
    IGNORED_DIRS = {
        '.git', 
        '__pycache__', 
        '.venv', 
        'node_modules', 
        'models', 
        'storage', 
        '.files',
        '.pytest_cache',
        '.ruff_cache'
    }
    
    tree_str = ""
    
    try:
        # Get all items in the directory, sorted
        items = sorted(os.listdir(root_dir))
    except PermissionError:
        return indent + " [Permission Denied]\n"

    # Filter out ignored directories
    items = [item for item in items if item not in IGNORED_DIRS]

    for i, item in enumerate(items):
        item_path = root_dir / item
        is_last = (i == len(items) - 1)
        
        # Determine prefix for the current line
        prefix = "└── " if is_last else "├── "
        tree_str += f"{indent}{prefix}{item}\n"
        
        if item_path.is_dir():
            # Determine indentation for the next level
            next_indent = indent + ("    " if is_last else "│   ")
            tree_str += generate_directory_tree(item_path, next_indent)
            
    return tree_str

def main():
    print("📁 --- Directory Structure Generator ---")
    
    # Ask user for path or use current directory as default
    input_path = input("\nEnter the path to analyze (press Enter for current project root): ").strip()
    
    if not input_path:
        root_path = Path.cwd()
    else:
        root_path = Path(input_path).resolve()
    
    if not root_path.exists() or not root_path.is_dir():
        print(f"❌ Error: Path '{root_path}' does not exist or is not a directory.")
        return

    print(f"🔍 Analyzing: {root_path}...")
    
    # Generate tree
    tree = f"Project Structure: {root_path.name}\n"
    tree += "=" * (len(root_path.name) + 19) + "\n\n"
    tree += root_path.name + "/\n"
    tree += generate_directory_tree(root_path)
    
    # Output file name
    output_file = "folder_structure.txt"
    output_path = root_path / output_file
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(tree)
        print(f"\n✅ Success! Structure saved to: {output_path}")
        print("\nPreview of root changes:")
        print("-" * 20)
        # Show first 15 lines of the tree
        print("\n".join(tree.splitlines()[:15]))
        print("...")
    except Exception as e:
        print(f"❌ Failed to save file: {e}")

if __name__ == "__main__":
    main()
