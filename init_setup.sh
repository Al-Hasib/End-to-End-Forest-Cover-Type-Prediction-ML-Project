#!/bin/bash

# Exit immediately if a command exits with a non-zero status
# set -e
echo [$(date)]: "START"

# Check for and install virtualenv if it's not installed
if ! [ -x "$(command -v virtualenv)" ]; then
  echo 'Error: virtualenv is not installed.' >&2
  echo 'Installing virtualenv...'
  pip install virtualenv
fi

# Create a virtual environment
if [ -d "venv" ]; then
  echo "Virtual environment already exists."
else
  echo "Creating virtual environment..."
  virtualenv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
  echo "Installing dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "requirements.txt not found."
fi

# Run any other initial setup commands
# For example, setting environment variables or running database migrations
# export FLASK_APP=app.py

echo "Setup complete."
