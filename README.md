# GOPH_419_Fall2025_Lab03_AU
Algorithm to solve second order ODE using Euler’s Method and classical 4th-order Runge- Kutta.
Installing: #clone the repository, #get into project root, #create and activate the virtual environment

'''bash

git clone 

cd GOPH_419_Fall2025_Lab03_AU

python -m venv .venv

.venv\Scripts\Activate

python -m pip install -r requirements.txt
#to create plots and run sensitivity test:
#plots are saved to GOPH_419_Fall2025_Lab03_AU/figures
python -m examples.driver
