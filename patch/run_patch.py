import os
import shutil
import argparse

def apply_patch(package_path, patch_filename):

    link_path = os.path.join(package_path, patch_filename)
    backup_path = os.path.join(package_path, f'{patch_filename}.bak')

    # Create a backup if it doesn't already exist
    if not os.path.exists(backup_path):
        if os.path.exists(link_path):
            shutil.move(link_path, backup_path)
            print(f"Created backup: {backup_path}")
        else:
            print("No existing file to back up.")
    else:
        print("Patch already installed. Skipping backup process.")

    # Remove the existing file or symlink
    if os.path.exists(link_path):
        if os.path.islink(link_path):
            os.unlink(link_path)
            print(f"Removed existing symlink: {link_path}")
        elif os.path.isfile(link_path):
            os.remove(link_path)
            print(f"Removed existing file: {link_path}")
        else:
            raise Exception(f"Existing path is not a file or symlink: {link_path}")

    # Get the current working directory and patch file path
    work_dir = os.path.abspath(os.getcwd())
    patch_path = os.path.join(work_dir, patch_filename)

    # Create the symlink
    try:
        # os.symlink(patch_path, link_path)
        # print(f"Created symlink: {link_path} -> {patch_path}")
        shutil.copy(patch_path, link_path)
        print(f"Created copy: {link_path} -> {patch_path}")
    except FileExistsError:
        print(f"Failed to create symlink: {link_path} already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--package_path', type=str, default=None, help='Path of the input data.')
    parser.add_argument('--patch_files', type=str, default=None, help='Path of the input data.')

    args = parser.parse_args()
    
    for patch_filename in args.patch_files.split(","):
        apply_patch(args.package_path, patch_filename)