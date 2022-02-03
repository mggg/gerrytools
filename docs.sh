
# Generate documentation with pdoc; rsync to the immediate parent directory;
# delete the old documentation directory.
pdoc evaltools --template-dir docs/templates --html --output-dir=docs --force
rsync -a docs/evaltools/ docs/
rm -rf docs/evaltools

# Get the current path to this script: borrowed from https://bit.ly/3vHDuR4.
SCRIPT_HOME=`dirname $0 | while read a; do cd $a && pwd && break; done`

# Serve the documentation.
python -m webbrowser -t "file://$SCRIPT_HOME/docs/index.html"
