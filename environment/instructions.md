Make sure you are in the root directory of the project.

1. Start Julia:
   ```
   julia
   ```
2. Open the package manager (Pkg mode) by pressing:
   ```
   ]
   ```
3. Activate the environment located in the environment/ folder:
   ```
   activate environment
   ```
4. Instantiate the environment to install all dependencies:
   ```
   instantiate
   ```
5. (Optional) Build any packages that require it:
   ```
   build
   ```
6. Exit Pkg mode by pressing backspace, and then julia with `exit()`.
7. From a terminal and in the root directory, execute the given `.sh` to open jupyter lab.
   ```
   . run_environment.sh
   ```
