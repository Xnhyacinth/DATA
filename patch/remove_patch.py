import os
import shutil
import argparse

def remove_patch(package_path, patch_filename):

    autoregressive_path = os.path.join(package_path, patch_filename)
    backup_path = os.path.join(package_path, f'{patch_filename}.bak')

    if os.path.exists(backup_path):

        if os.path.exists(autoregressive_path):
            os.remove(autoregressive_path)
            print(f"Removed existing file: {autoregressive_path}")

        shutil.move(backup_path, autoregressive_path)
        print(f"Restored backup: {autoregressive_path}")

    link_path = os.path.join(package_path, patch_filename)

    if os.path.islink(link_path):
        os.remove(link_path)
        print(f"Removed symlink: {link_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--package_path', type=str, default=None, help='Path of the input data.')
    parser.add_argument('--patch_files', type=str, default=None, help='Path of the input data.')
    args = parser.parse_args()
    
    for patch_filename in args.patch_files.split(","):
        remove_patch(args.package_path, patch_filename)