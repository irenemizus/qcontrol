# Preparing environment
Here we put all the commands necessary to create an Anaconda/Python 3 virtual environment for the project.

Tested on macOS Monterey 12.6 with Anaconda 4.12.0

```
> conda create -n newcheb numpy==1.23.1
> conda activate newcheb
> pip install jsonpath2==0.4.5
```

Every time you need to use the application, activate the environment with:
```
conda activate newcheb
```
