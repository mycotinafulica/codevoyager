# Setting Up Environment

To set up your environment, you need an anaconda installed in your machine. Then create the conda environment by running the following command (make sure that conda is within your path): 

```sh
conda env create -f environment.yml
```

This project will leverage Open AI API, please create .env file and have the following entry:
```
OPENAI_API_KEY=[YOUR_API_KEY_HERE]
```

# Activating the Environment

Once your environment is set up, activate your, you can activate your environment by running the following command : 

```sh
conda activate codevoyager
```

# Editing Environment

Shall you have the need to edit the environment dependency, run the following commands:

```sh
conda activate myenv
conda env update --file environment.yml --prune
```

# Run The Web UI
After activating your conda environment, run:
```
python run.py
```

Then open the link into your browser.