#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
import numpy as np
import datetime
from numbers import Number


# any to datetime.datetime
ANY2DT = Union[datetime.datetime,
               datetime.date,
               np.datetime64,
               str]

# any to datetime.timedelta
ANY2TD = Union[datetime.timedelta,
               np.timedelta64,
               str,
               Number]

INT_FLOAT_ANY2DT = Union[int, float, ANY2DT]
