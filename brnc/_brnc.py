#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

__all__ = ["index_of_valid_value_along_axis",
           "valid_value_along_axis",
           "length_to_slices_of_indexes"]

# %reset -f

from typing import Union, Optional, Iterator
from functools import singledispatch
import itertools
import datetime
import time
import numpy as np
import pandas as pd
import xarray as xr

import logging

# use GMT/Zulu time
logging.Formatter.converter = time.gmtime
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%SZ",
                    force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

ANY2DATETIME = Union[datetime.datetime,
                     datetime.date,
                     np.datetime64,
                     str]

INT_FLOAT = Union[int, float]

INT_FLOAT_DT64 = Union[int, float, np.datetime64]

DX = Union[xr.DataArray, xr.Dataset]


def index_of_valid_value_along_axis(arr: np.ndarray,
                                    *,
                                    axis: int,
                                    position: str = "first",
                                    ) -> np.ndarray:
    """
    Get the index of the first or last valid value along a specific axis of an
    input array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array from which to retrieve the index of the first or last valid
        value.
    axis : int
        Axis along which to search for the valid value.
    position : str, optional
        Specifies whether to retrieve the index of the first or last valid value
        along the axis. Default is "first".

    Returns
    -------
    numpy.ndarray
        Array containing the index of the first or last valid value along the
        specified axis. -1 is returned if all elements along the axis are NaN.

    Examples
    --------
    >>> arr = np.array([[np.nan, 2, 5],
                        [7, 3, 8],
                        [0, 4, np.nan],
                        [np.nan, np.nan, np.nan]])

    >>> index_of_valid_value_along_axis(arr, axis=0, position="first")
    array(1, 0, 0)

    >>> index_of_valid_value_along_axis(arr, axis=1, position="last")
    array([ 2,  2,  1, -1])

    """

    _ = int(str(axis))

    ones = np.ones(arr.shape)

    valid = np.where(~np.isnan(arr), ones, np.nan)

    valid_indexes = ((ones.cumsum(axis=axis) - 1) * valid)

    positions = {"first": np.nanmin, "last": np.nanmax}

    indexes = positions[position](valid_indexes, axis=axis)

    # use -1 if all elements along axis are nan
    # Note: -1 is a valid index when searching the array, so be careful
    indexes = np.where(np.isnan(indexes), -1, indexes).astype(int)

    return indexes


def valid_value_along_axis(arr: np.ndarray,
                           *,
                           axis: int,
                           position: str = "first",
                           ) -> np.ndarray:
    """
    Get the first or last valid value along a specific axis of an input array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array from which to retrieve valid values.
    axis : int
        Axis along which to retrieve valid values.
    position : str, optional
        Specifies whether to retrieve the first or last valid value along the axis.
        Default is "first".

    Returns
    -------
    numpy.ndarray
        Array containing the first or last valid value along the specified axis.
        NaN is returned if all elements along the axis are NaN.

    Examples
    --------
    >>> arr = np.array([[np.nan, 2, 5],
                        [7, 3, 8],
                        [0, 4, np.nan],
                        [np.nan, np.nan, np.nan]])

    >>> valid_value_along_axis(arr, axis=0, position="first")
    array([7., 2., 5.])

    >>> valid_value_along_axis(arr, axis=1, position="last")
    array([ 5.,  8.,  4., nan])

    """

    _ = int(str(axis))

    indexes = index_of_valid_value_along_axis(arr,
                                              axis=axis,
                                              position=position)

    other_axes = np.ogrid[tuple([slice(size) for size in indexes.shape])]
    # other_axes = np.meshgrid(*[range(size) for size in indexes.shape],
    #                          sparse=True,
    #                          indexing="ij")

    other_axes.insert(axis, indexes)

    return np.where(indexes < 0, np.nan, arr[tuple(other_axes)])


def length_to_slices_of_indexes(length: int, step: int
                                ) -> Iterator[slice]:
    """
    Convert length to slices of indexes using step.

    Parameters
    ----------
    length : int
        The total length to convert into slices of indexes.
    step : int
        The step for each slice.

    Yields
    ------
    slice
        A slice representing the indexes based on the specified step.

    Examples
    --------
    >>> for s in length_to_slices_of_indexes(14, 4):
    ...     print(s)
    slice(0, 4, None)
    slice(4, 8, None)
    slice(8, 12, None)
    slice(12, 14, None)

    >>> list(length_to_slices_of_indexes(14, 4))
    [slice(0, 4, None), slice(4, 8, None), slice(8, 12, None), slice(12, 14, None)]

    """

    _ = int(str(length))
    _ = int(str(step))

    for arr in np.array_split(range(length), range(step, length, step)):
        yield slice(arr[0], arr[-1] + 1)


def shape2chunk(*,
                shape: tuple[int, ...],
                size: int,
                preferred_axes: Optional[list] = None,
                preferred_in_order: bool = False
                ) -> tuple[int, ...]:
    """
    Calculate how to best split a array with shape `shape` in smaller chunks
    so that the number of elements in each chunk is closest and less or equal
    to `size`.

    Parameters
    ----------
    shape : tuple
        The shape of a np.ndarray, e.g.: `arr.shape`
    size : int
        Maximum number of elements in the resulting chunk, e.g.:
        `chunk[0] * chunk[1] * ... chunk[N] <= size`
    preferred_axes : list, optional
        A list of preferred axes for chunking, by default None.
    preferred_in_order : bool, optional
        Specifies if the preferred axes should be in order, by default False.

    Returns
    -------
    chunk : tuple
        Chunk shape.

    Notes
    -----
    This function attempts to find a "balanced" chunk shape, not simply the one
    closest to size, e.g.: for a array with shape (1000, 30, 200, 100) and
    desired chunk size 1024, a balanced chunk shape is (6, 6, 5, 5) with 900
    elements, while the closest solution would be an unbalanced (1024, 1, 1, 1).

    Examples
    --------
    >>> shape2chunk(shape=(1000, 30, 200, 100), size=1024)
    (6, 6, 5, 5)

    >>> shape2chunk(shape=(1000, 30, 200, 100), size=1024,
    ...             preferred_axes=[0, 1])
    (34, 30, 1, 1)

    >>> shape2chunk(shape=(1000, 30, 200, 100), size=1024,
    ...             preferred_axes=[0, 1], preferred_in_order=True)
    (1000, 1, 1, 1)

    """

    _ = [int(str(size)) for size in shape]
    _ = int(str(size))

    if preferred_axes:
        preferred_axes = ([[axis] for axis in preferred_axes]
                          if preferred_in_order
                          else [preferred_axes])

    else:
        preferred_axes = []

    rngs = [range(1, size) for size in shape]

    for axes in preferred_axes:

        rngs2 = [rng if axis in axes else range(rng.start, rng.start)
                 for axis, rng in enumerate(rngs)]

        chunks = _ranges2product(rngs2, size)

        rngs = [range(chunks[axis], chunks[axis]) if axis in axes else rng
                for axis, rng in enumerate(rngs)]

    return _ranges2product(rngs, size)


def _ranges2product(rngs: list[range],
                    prod: int
                    ) -> tuple[int, ...]:
    """
    Find the values in a list of ranges so that the product between the values
    is closest and less or equall than prod.

    Parameters
    ----------
    rngs : list
        A list of range objects.
    prod : int
        The target product value.

    Returns
    -------
    out : tuple
        Tuple with the value in each range so that the product between the
        values is closest and less or equall than prod.

    Examples
    --------
    >>> rngs = range(1, 10), range(2, 15), range(4, 20)
    >>> _ranges2product(rngs, 102)
    (5, 5, 4)

    """

    starts = list(map(lambda x: x.start, rngs))
    stops = list(map(lambda x: x.stop, rngs))
    max_size = max(stops)

    if np.r_[starts].prod() > prod:
        raise ValueError(
            "The smaller product between the ranges is greater than prod")

    arr = np.ones((len(stops), max_size)).cumsum(axis=1).astype(int).T

    arr = np.clip(arr, a_min=starts, a_max=stops)

    delta = arr.prod(axis=1) - prod

    idx = np.nanargmin(np.where(delta > 0, np.nan, np.abs(delta)))

    # do +1 to one or more columns to get even closer to prod
    zerone = np.array(list(itertools.product([True, False],
                                             repeat=len(stops)))).astype("int")

    arr = np.clip(arr[idx, :] + zerone, a_min=starts, a_max=stops)
    arr = arr[np.argsort(arr.prod(axis=1))]

    delta = arr.prod(axis=1) - prod
    idx = np.nanargmin(np.where(delta > 0, np.nan, np.abs(delta)))

    return tuple(arr[idx, :])


@singledispatch
def any2datetime(dt: ANY2DATETIME,
                 dt_fmt: Optional[str] = None) -> datetime.datetime:
    """Convert `dt` to datetime.datetime object.

    Parameters
    ----------
    dt : datetime.datetime, datetime.date, np.datetime64, str

        * datetime.datetime: just returns
        * datetime.date: convert to datetime.datetime with time 00:00:00.
        * np.datetime64: convert to datetime.datetime.
        * str: tries to parse the string assuming a standard format. If
            `dt_fmt`, tries to parse the string using `dt_fmt` as format.

    dt_fmt : str, optional
        `dt` format. Only used if `dt` is str.
        Examples: '%Y-%m-%d %H:%M:%S', 'd=%d m=%m a=%Y'

    Returns
    -------
    dt : datetime.datetime

    Examples
    --------
    >>> any2datetime(datetime.datetime(1983, 12, 19))
    datetime.datetime(1983, 12, 19, 0, 0)

    >>> any2datetime(datetime.date(1983, 12, 19))
    datetime.datetime(1983, 12, 19, 0, 0)

    >>> any2datetime(np.datetime64("1983-12-19"))
    datetime.datetime(1983, 12, 19, 0, 0)

    >>> any2datetime("19831219")
    datetime.datetime(1983, 12, 19, 0, 0)

    >>> any2datetime("01/02/2010")   # assumes day first dd/mm/yyyy
    datetime.datetime(2010, 2, 1, 0, 0)

    >>> any2datetime("1983-12-19T21:35:57-03:00")
    datetime.datetime(1983, 12, 19, 21, 35, 57, tzinfo=pytz.FixedOffset(-180))

    >>> any2datetime("1983-12-19T21:35:57Z")
    datetime.datetime(1983, 12, 19, 21, 35, 57, tzinfo=<UTC>)

    >>> any2datetime("d=19 m=12 a=1983")
    ParserError: Unknown string format: d=19 m=12 a=1983 present at position 0

    >>> any2datetime("d=19 m=12 a=1983", "d=%d m=%m a=%Y")
    datetime.datetime(1983, 12, 19, 0, 0)

    """
    raise TypeError("Invalid type!")


@any2datetime.register(datetime.datetime)
def _(dt: datetime.datetime) -> datetime.datetime:
    return dt


@any2datetime.register(datetime.date)
def _(dt: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(dt, datetime.datetime.min.time())


@any2datetime.register(np.datetime64)
def _(dt: np.datetime64) -> datetime.datetime:
    return pd.to_datetime(dt).to_pydatetime()


@any2datetime.register(str)
def _(dt: str, dt_fmt: Optional[str] = None) -> datetime.datetime:

    if dt_fmt is not None:
        return datetime.datetime.strptime(dt, dt_fmt)

    # using pandas parser instead of dateutil because of
    # https://github.com/dateutil/dateutil/issues/402
    #
    # >>> pd.to_datetime('02/03/1983', dayfirst=True).to_pydatetime()
    # datetime.datetime(1983, 3, 2, 0, 0) -> OK
    # >>> dateutil.parser.parse('02/03/1983', dayfirst=True)
    # datetime.datetime(1983, 3, 2, 0, 0) -> OK
    #
    # >>> pd.to_datetime('19830302', dayfirst=True).to_pydatetime()
    # datetime.datetime(1983, 3, 2, 0, 0)
    # >>> dateutil.parser.parse('19830302', dayfirst=True)
    # datetime.datetime(1983, 2, 3, 0, 0) -> NOT OK
    #
    # use dayfirst=True and yearfirst=False for 2 digits year (%y)
    #
    # >>> pd.to_datetime('10/09/08', dayfirst=True).to_pydatetime()
    # datetime.datetime(2008, 9, 10, 0, 0) -> OK
    # >>> pd.to_datetime('10/09/08', dayfirst=True, yearfirst=True).to_pydatetime()
    # datetime.datetime(2010, 8, 9, 0, 0) -> NOT OK
    return pd.to_datetime(dt, dayfirst=True).to_pydatetime()


@xr.register_dataarray_accessor("br")
class BrDA:

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def da(self) -> xr.DataArray:
        return self._obj

    @property
    def name(self) -> str:
        return getattr(self.da, "name", "unnamed")

    def err(self, msg: str) -> None:
        """Raise a ValueError with message msg."""
        raise ValueError(f"DataArray '{self.name}': " + msg)

    def info(self, msg: str) -> None:
        """Info message msg."""
        log.info(f"DataArray '{self.name}': " + msg)

    def warn(self, msg: str) -> None:
        """Warning message msg."""
        log.warning(f"DataArray '{self.name}': " + msg)

    def load(self) -> xr.DataArray:

        if self.da._in_memory:
            return self.da

        self.info("loading data in memory")

        return self.da.load()

    @property
    def is_numeric(self) -> bool:
        """
        Check whether the DataArray is of a numeric dtype.

        Returns
        -------
        boolean
            Whether or not the array or dtype is of a numeric dtype.

        """
        return pd.api.types.is_numeric_dtype(self.da)

    def index_of_valid_value_along_dimension(self,
                                             dim: str,
                                             position: str = "first",
                                             ) -> xr.DataArray:
        """
        Get the index of the first or last valid value along a specific
        dimension.

        Parameters
        ----------
        dim : str, optional
            The name of the dimension along which to get the index.
        position : str, optional
            Specifies whether to retrieve the index of the first or last valid
            value along the dimension or axis.
            Default is "first".

        Returns
        -------
        xr.DataArray
            xr.DataArray containing the index of the first or last valid value
            along the specified dimension or axis.

        """

        # Note: if da is a <class 'dask.array.core.Array'>, the code fails with:
        # "NotImplementedError: Don't yet support nd fancy indexing"
        # so call load to make sure da is a np.array with data in memory.

        return self.da.br.load().reduce(
            index_of_valid_value_along_axis,
            dim=dim,
            keep_attrs=True,
            position=position)

    def valid_value_along_dimension(self,
                                    dim: str,
                                    position: str = "first",
                                    ) -> xr.DataArray:
        """
        Get the first or last valid value along a specific dimension.

        Parameters
        ----------
        dim : str, optional
            The name of the dimension along which to get the value.
        position : str, optional
            Specifies whether to retrieve the first or last valid value along
            the dimension or axis.
            Default is "first".

        Returns
        -------
        xr.DataArray
            xr.DataArray containing the first or last valid value along the
            specified dimension or axis.

        """

        # Note: if da is a <class 'dask.array.core.Array'>, the code fails with:
        # "NotImplementedError: Don't yet support nd fancy indexing"
        # so call load to make sure da is a np.array with data in memory.

        return self.da.br.load().reduce(
            valid_value_along_axis,
            dim=dim,
            keep_attrs=True,
            position=position)
