from glob import glob
import numpy as np

import rasterio
import json, re, itertools, os

import matplotlib.pyplot as plt

import cv2 as cv
from sklearn import preprocessing
from sklearn.cluster import KMeans

N_OPTICS_BANDS = 7

with open("bands.json","r") as bandsJson:
    bandsCharacteristics = json.load(bandsJson)

    