# Marugoto Installation and Recommended Project Setup

# Installing and setting up marugoto

1. clone the marugoto directory from github to your local folder(referred to as $marugoto = ~/Documents/) via:
    
    ```bash
    git clone https://github.com/KatherLab/marugoto
    ```
    
2. set up your local virtual environment in your (folder referenced to as $env = ~/Documents/venvs/) 
    
    `python -m venv venv_name`
    
3. Activate your environment 
    
    `source $env/venv_name/bin/activate`
    
4. To find out which cuda version is appropriate, run `nvidia-smi` to view the cuda version in the top right-hand corner:
    
    ![Untitled](https://i.imgur.com/hfDViM2.png)
    
5. [Install Pytorch](https://pytorch.org/get-started/locally/) by using the command provided by the website after selecting the appropriate settings.
6. Confirm cuda is installed correctly by `python` and then in shell
    
    ```python
    import torch
    torch.cuda.is_available()
    ```
    
    This should return `True`
    
7. Go to `$marugoto/marugoto/` and do
    
    `pip install .`
    
8. marugoto should now be correctly installed and ready for use.

# Project management

1. If your OS is installed on a smaller local drive with data stored on larger mounted drives, we recommend storing project data between the two drive as follows. If only a single drive is available, then all data will need to be stored there.
    
    
    | local drive | mounted HDD |
    | --- | --- |
    | marugoto | WSI’s |
    | Scripts | Tiles |
    | Results | Features |
2. A good structure is to have a folder for each main project documents with a symbolic link, scripts and experiment results located within it (see below). To create a symbolic link, using the terminal when inside the project folder, run `ln -s ~/Documents/marugoto/marugoto`

    ![Untitled](https://i.imgur.com/0TITTqA.png)
