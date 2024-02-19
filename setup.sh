######################
# CONTRIBUTING SETUP #
######################

# Run this file to set up your environment to contribute to gerrytools.
# You will need to run this from the root of the gerrytools repository.
echo "Making python virtual environment '.venv' in current directory"
python3 -m venv .venv
echo "Environment created"
# Activate the environment for macOS and Linux
if [ "$(uname)" == "Darwin" ] || [ "$(uname)" == "Linux" ]; then
    printf "\n\nActivating environment"
    source .venv/bin/activate
# Activate the environment for Windows
else
    printf "\n\nActivating environment"
    source .venv/Scripts/activate
fi
echo "Environment activated"

echo "Updating pip"
pip install -qq --upgrade pip
echo "pip has been updated"

printf "\n\nInstalling requirements from requirements.txt into environment\n"
pip install -qq -r requirements.txt
echo "Done!"

printf "\n\nInstalling working directory's version of gerrytools into environment\n"
pip install -qq -e .
echo "Done!"

printf "\n\nInstalling pre-commit hooks\n"
pip install -qq pre-commit
pre-commit autoupdate
pre-commit install
echo "Done!"

printf "\n\nSetup complete! You are now ready to contribute to gerrytools.\n"