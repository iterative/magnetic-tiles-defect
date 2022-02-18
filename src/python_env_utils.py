import pkg_resources


def write_pkg_list_to_file(fpath):
    packages = pkg_resources.working_set
    packages_list = sorted([f"{i.key}=={i.version}" for i in packages])
    with open(fpath, 'w') as f:
        for item in packages_list:
            f.write(f"{item}\n")
