<!-- Needs changes -->

![screenshot](docs/slides/Slide2.jpg)
![screenshot](docs/slides/Slide3.jpg)
![screenshot](docs/slides/Slide4.jpg)
![screenshot](docs/slides/Slide5.jpg)
![screenshot](docs/slides/Slide6.jpg)
![screenshot](docs/slides/Slide7.jpg)
![screenshot](docs/slides/Slide8.jpg)
![screenshot](docs/slides/Slide9.jpg)
![screenshot](docs/slides/Slide10.jpg)
![screenshot](docs/slides/Slide11.jpg)
![screenshot](docs/slides/Slide12.jpg)
![screenshot](docs/slides/Slide13.jpg)
![screenshot](docs/slides/Slide14.jpg)
![screenshot](docs/slides/Slide15.jpg)
![screenshot](docs/slides/Slide16.jpg)
![screenshot](docs/slides/Slide17.jpg)
![screenshot](docs/slides/Slide18.jpg)
![screenshot](docs/slides/Slide19.jpg)
![screenshot](docs/slides/Slide20.jpg)
![screenshot](docs/slides/Slide21.jpg)
![screenshot](docs/slides/Slide22.jpg)
![screenshot](docs/slides/Slide23.jpg)
![screenshot](docs/slides/Slide24.jpg)



Python env setup
load python 3.8 module
in SLURM - module load python/3.8.18 (verify the version using "module avail" command)

<!-- to ensure virtual environments are created in the local directory -->
use "poetry config virtualenvs.in-project true"
to install - "poetry install"

activate virtual environment created by poetry in the project directory
source .venv/bin/activate

<!-- to be able to install torch-scatter, open issue at https://github.com/rusty1s/pytorch_sparse/issues/156 -->
<!-- please refer here for different options -->
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

poetry config repositories.torch-wheels https://data.pyg.org/whl/torch-2.2.0+cpu.html

poetry source add torch-wheels https://data.pyg.org/whl/torch-2.2.0+cpu.html

if using slurm jobs, create directory for logs using mkdir logs


jupyterlab setup details can be found at https://jejjohnson.github.io/research_journal/tutorials/remote_computing/vscode_jlab/

docs setup
---------------
mkdir docs
cd docs
sphinx-quickstart
-- update conf.py

from project directory
sphinx-apidoc -o docs/source/ graphxl/
sphinx-build -b html docs/source docs/build

from docs directory
./make.sh html

from docs/build directory
python -m http.server to run the docs server locally

