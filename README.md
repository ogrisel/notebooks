# Sample IPython 0.12+ notebooks for machine learning stuff

## Screenshots

![Digits](https://github.com/ogrisel/notebooks/raw/master/screenshots/digits.png)
![Topics](https://github.com/ogrisel/notebooks/raw/master/screenshots/topics.png)

## Install dependencies

    sudo pip install -U tornado
    sudo pip install -U pyzmq
    sudo pip install -U git+https://github.com/ipython/ipython.git

## Run the notebook

Then `cd` into this folder and run:

    ipython notebook --pylab=inline

Then click on a notebook. The focus is on the first cell: hit
`Shift-Enter` to execute the current cell an move on to the next. You
can also click on `Run > All` in the left panel.
