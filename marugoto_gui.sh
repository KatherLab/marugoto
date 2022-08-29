#!/bin/bash
python3 -m venv /home/$USER/.virtualenvs/marugoto 
source /home/$USER/.virtualenvs/marugoto/bin/activate
git pull
python3 -m pip install --upgrade pip
pip3 install .
pip3 install openpyxl
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
python3 marugoto/gui/marugoto_gui.py
