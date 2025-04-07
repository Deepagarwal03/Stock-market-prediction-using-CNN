# cnn-stock-prediction
Sample code for using 1-dimensional Convolutional Neural Networks (CNNs) to predict stock price movements
Installation
------------

Install dependencies using ``pip``::

    pip install -r requirements.txt

If you have CUDA installed, you may want to install [pytorch](https://pytorch.org/) separately. Doing so will
significantly speed up model training.

Running
------------

The following will automatically download the data, train the model, and generate charts comparing forecasts
to actual.

    python main.py


