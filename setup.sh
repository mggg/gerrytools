
######################
# CONTRIBUTING SETUP #
######################

# Run this file to set up your environment to contribute to gerrytools.
echo "installing pre-commit documentation hook to .git/hooks/pre-commit"
curl https://evaltools-test-data.s3.amazonaws.com/pre-commit --output .git/hooks/pre-commit
chmod +x ./git/hooks/pre-commit

echo "\n\ninstalling pre-push linting hook to .git/hooks/pre-push"
curl https://evaltools-test-data.s3.amazonaws.com/pre-push --output .git/hooks/pre-push
chmod +x ./git/hooks/pre-push

echo "\n\ninstalling flake8 configuration to .flake8"
curl https://evaltools-test-data.s3.amazonaws.com/.flake8 --output .flake8
