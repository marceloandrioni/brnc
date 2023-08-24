#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
import numpy as np
import datetime


ANY2DT = Union[datetime.datetime,
               datetime.date,
               np.datetime64,
               str]

INT_FLOAT = Union[int, float]

INT_FLOAT_DT64 = Union[int, float, np.datetime64]

INT_FLOAT_ANY2DT = Union[int, float, ANY2DT]
