#!/usr/bin/env bash

# Move into the ML1 folder (directory of the script)
# 1. Find the folder containing the script (dirname)
# 2. Change to that folder temporarily (cd)
# 3. Get the absolute path to it (pwd)
# 4. Assign that path to SCRIPT_DIR
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Starting Julia environment from: $SCRIPT_DIR"

# Run Julia and start JupyterLab with the ML1 environment active
julia --project=environment -e '
using IJulia;
println("Launching JupyterLab with ML1 as root...");
jupyterlab(dir="'$SCRIPT_DIR'");
'

# using Pkg;
# Pkg.activate("environment");
