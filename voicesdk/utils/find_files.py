from glob import glob


def find_files_recursive(directory, extension=".wav"):
    if not extension.startswith('.'):
        extension = '.' + extension

    pattern = f"{directory}/**/*{extension}"
    files = glob(pattern, recursive=True)

    return files
