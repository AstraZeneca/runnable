import sys

from tomlkit import dump, load

if len(sys.argv) < 2:
    msg = (
        "Usage: python upload_version <new_version> <output_to, defaults to pyproject.toml>"
    )

new_version = sys.argv[1]
output_file = "pyproject.toml"
if len(sys.argv) > 2:
    output_file = sys.argv[2]

toml_file = "pyproject.toml"
contents = load(open(toml_file, 'rb'))
contents["tool"]["poetry"]["version"] = str(new_version)
dump(contents, open(output_file, 'w'))
