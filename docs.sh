pdoc evaltools --html --output-dir=docs --force
rsync -a docs/evaltools/ docs/
rm -rf docs/evaltools