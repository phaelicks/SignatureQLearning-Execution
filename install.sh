#!/bin/bash
# Ensure we use Python 3.8 venv
source venv/bin/activate

# Force pip version 21.3.1
python3 -m pip install --upgrade pip==21.3.1 setuptools==44.1.1 wheel==0.34.2

# Install requirements
python3 -m pip install -r requirements.txt

# Install the ABIDES packages
cd abides-jpmc-public/abides-core
python3 setup.py install
cd ../abides-markets
python3 setup.py install
cd ../abides-gym
python3 setup.py install
cd ../../
