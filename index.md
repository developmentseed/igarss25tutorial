# Harnessing the Power of Earth Foundation Models

Welcome to the IGARSS25 EarthFM tutorial 🥳! This Jupyter book 📖 contains notebooks on
how to apply 🧱 Foundation Models to 🛰️ Earth Observation data.

## 📚 Overview of tutorials

1. Introduction to Earth Foundation Models
2. Evaluating EO Foundation Models
3. Applying Foundation Models to Segmentation downstream task

Each tutorial is rendered on this website for easy viewing 👀, but some of them are
Jupyter notebooks designed to be ran interactively 💫. See the instructions below on how
you can start running the tutorials in no time! 🚀

## 🌠 Setting up your environment

To run these notebooks in an interactive Jupyter session online, 🖱️ click on the button
below to launch on
[Google Colaboratory](https://colab.google).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/developmentseed/igarss25tutorial/blob/main/tut2_EOFM_Evaluation.ipynb)

Alternatively, you can choose to run the Jupyter notebooks on another cloud provider
with GPU instances such as [Sagemaker Studio Lab](https://studiolab.sagemaker.aws).

[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/developmentseed/igarss25tutorial/blob/main/tut2_EOFM_Evaluation.ipynb)

### 💻 Creating a local environment for running tutorials

If you prefer to run the 🧑‍🏫 tutorials with a local installation instead, then follow
along! For this IGARSS25 workshop, we recommend creating a virtual environment with
[`uv`](https://docs.astral.sh/uv) and installing the 🐍 Python libraries inside.

:::{tip}
For users comfortable with using `git`, feel free to ⬇️ download or clone the repository
containing the tutorial materials directly using
```bash
git clone https://github.com/developmentseed/igarss25tutorial.git
```
:::

Here's the instructions to install the `igarss25tutorial` environment:

1. Ensure that you have the
   [`uv`](https://docs.astral.sh/uv/getting-started/installation) package manager
   installed.

2. Make a folder called 'igarss25tutorial'. This will be where you will put all the
   Jupyter notebooks and data files 🗃️ used in the workshop.

3. Download a copy of the 'pyproject.toml' and 'uv.lock' files which contains a 📄 list
   of dependencies required to run the tutorials in this workshop. Get it at
   https://github.com/developmentseed/igarss25tutorial/blob/main/pyproject.toml and
   https://github.com/developmentseed/igarss25tutorial/blob/main/uv.lock

4. Run the following commands on the 🧑‍💻 command-line to create the virtual environment

   ```bash
   cd /path/to/igarss25tutorial
   uv sync --locked
   ```

5. Once the installation is completed 🏁, launch
   [Jupyter Lab](https://jupyterlab.readthedocs.io) as follows:

    ```bash
    source .venv/bin/activate
    uv run --with jupyter jupyter lab
    ```

   This should open up a page in your default browser. If not, you can click and open
   the 🔗 link that says `http://localhost:8888/lab?token=...` in your command-line
   terminal and this will take you to the Jupyter Lab page.

6. Download the Jupyter notebook(s) you want to run (e.g.
   https://www.developmentseed.org/igarss25tutorial/tut2-eofm-evaluation/) using
   either the download button on the ↗️ top right (select '.ipynb') or from GitHub at
   https://github.com/developmentseed/igarss25tutorial. Make sure to put
   the \*.ipynb file(s) inside of the 'igarss25tutorial' folder.

7. Open the Jupyter notebook in the left-pane file browser, e.g. by 🖱️ double-clicking
   on `tut2_EOFM_Evaluation.ipynb`. You are now ready to run through the course materials 🎉!


```{admonition} Acknowledgements
The contents of the first two tutorial pages are derived from a draft chapter on
Foundation Models that will become a part of
[the SERVIR Applied Deep Learning Book](https://servir.github.io/SERVIR-Applied-Deep-Learning-Book).
We would like to thank [SERVIR](https://servirglobal.net) for their generous permission
to re-purpose the contents of the book chapter for this IGARSS 2025 tutorial.
```
