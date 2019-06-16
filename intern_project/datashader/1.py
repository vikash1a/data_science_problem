# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:05:51 2019

@author: vikash
"""

import datetime
import pandas as pd
import numpy as np
import xarray as xr
import datashader as ds
import datashader.transfer_functions as tf
from collections import OrderedDict

# =============================================================================
# Create some fake timeseries data
# =============================================================================
# Constants
np.random.seed(2)
n = 100000                               # Number of points
cols = list('abcdefg')                   # Column names of samples
start = datetime.datetime(2010, 10, 1, 0)   # Start time

# Generate a fake signal
signal = np.random.normal(0, 0.3, size=n).cumsum() + 50

# Generate many noisy samples from the signal
noise = lambda var, bias, n: np.random.normal(bias, var, n)
data = {c: signal + noise(1, 10*(np.random.random() - 0.5), n) for c in cols}

# Add some "rogue lines" that differ from the rest 
cols += ['x'] ; data['x'] = signal + np.random.normal(0, 0.02, size=n).cumsum() # Gradually diverges
cols += ['y'] ; data['y'] = signal + noise(1, 20*(np.random.random() - 0.5), n) # Much noisier
cols += ['z'] ; data['z'] = signal # No noise at all

# Pick a few samples from the first line and really blow them out
locs = np.random.choice(n, 10)
data['a'][locs] *= 2

# Create a dataframe
data['Time'] = [start + datetime.timedelta(minutes=1)*i for i in range(n)]

df = pd.DataFrame(data)
df.tail()


df['ITime'] = pd.to_numeric(df['Time']).astype(int)

# Default plot ranges:
x_range = (df.iloc[0].ITime, df.iloc[-1].ITime)
y_range = (1.2*signal.min(), 1.2*signal.max())

print("x_range: {0} y_range: {0}".format(x_range,y_range))

cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=300, plot_width=900)
aggs= OrderedDict((c, cvs.line(df, 'ITime', c)) for c in cols)
img = tf.shade(aggs['a'])


img

mask = (df.index % 10) == 0
tf.shade(cvs.line(df[mask][['a','ITime']], 'ITime', 'a'))

renamed = [aggs[key].rename({key: 'value'}) for key in aggs]
merged = xr.concat(renamed, 'cols')
tf.shade(merged.any(dim='cols'))

colors = ["red", "grey", "black", "purple", "pink",
          "yellow", "brown", "green", "orange", "blue"]
imgs = [tf.shade(aggs[i], cmap=[c]) for i, c in zip(cols, colors)]
tf.stack(*imgs)

tf.stack(*reversed(imgs))

total = tf.shade(merged.sum(dim='cols'), how='linear')
total

tf.stack(total, tf.shade(aggs['z'], cmap=["lightblue", "red"]))

tf.stack(total, tf.shade(aggs['y'], cmap=["lightblue", "red"]))


cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=300, plot_width=900)
agg = cvs.area(df, x='ITime', y='a')
img = tf.shade(agg)
img

import holoviews as hv
from holoviews.operation.datashader import datashade
hv.extension('bokeh')

opts = hv.opts.RGB(width=600, height=300)
ndoverlay = hv.NdOverlay({c:hv.Curve((df['Time'], df[c]), kdims=['Time'], vdims=['Value']) for c in cols})
datashade(ndoverlay, normalization='linear', aggregator=ds.count()).opts(opts)