#!/bin/bash

echo "**********************"
echo "* generating docs... *"
echo "**********************"

# Generate documentation with pdoc; rsync to the immediate parent directory;
# delete the old documentation directory.
pdoc gerrytools --template-dir docs/templates --html --output-dir=docs --force
rsync -a docs/gerrytools/ docs/
rm -rf docs/gerrytools


echo "**********************"
echo "* complete! pushing. *"
echo "**********************"
