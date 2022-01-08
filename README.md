# sddip
Implementation of the stochastic dual dynamic integer programming (SDDiP) algorithm with dynamic binarization.

## Setup
Before the contents of this project can be explored and used smoothly, a few steps are required to set it up. The central part of the project is located in the Python package named `sddip`. For absolute imports like `import sddip` to work, the package must be introduced to python's lookup procedure. A way to do this is to either `pip install` the package or to add the directory `C:\path\to\sddip\sddip` to the environment variable called `PYTHONPATH`. The latter is convenient, as it allows the package to be easily edited and applied without reinstalling it. For detailed instructions on how to add the package directory to the path variables via your virtual environment follow this [link](https://towardsdatascience.com/how-to-fix-modulenotfounderror-and-importerror-248ce5b69b1c). Alternatively, follow the guide on how to set the required environment variable on windows under this [link](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-so-it-finds-my-modules-packages).
