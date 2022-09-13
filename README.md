# Preparing environment (journal)
Here we put all the commands necessary to create a Python 3 virtual environment for the project.

Tested on pure Ubuntu 20.04.2, Python 3.8.5

```sudo apt install python3-venv```

Change dir to `newcheb`.

```python3 -m venv venv```

The Virtual Environment is created in a folder `venv`

Then the newly created environment should be activated by

```source venv/bin/activate```

Then as we are inside the env, let's install the required Python packages into it.

```pip install --upgrade pip setuptools```

```pip install -r requirements.txt```

# Activating the environment

Go to the `newcheb` folder

```source venv/bin/activate```