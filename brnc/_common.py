#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections.abc import Iterable
from typing import Union, Iterator, Optional
from functools import singledispatch
from itertools import accumulate
import re
import datetime
import numpy as np
import pandas as pd
import itertools


def number2int(x: Union[int, float]) -> int:
    """
    Convert number to an integer without precision loss.

    Parameters
    ----------
    x : int, float
        The number to be converted.

    Returns
    -------
    out : int
        The converted number as an integer.

    Examples
    --------
    >>> number2int(10)
    10

    >>> number2int(3.14)
    ValueError: Number 3.14 can't be cast to int without precision loss

    """

    if x - int(x):
        raise ValueError("Number {x} can't be cast to int without precision loss")
    return int(x)


def parse_file_size(size: str) -> int:
    """
    Convert a file size in human-readable format to bytes.

    Parameters
    ----------
    size : str
        The file size in human-readable format, e.g., "10KB", "2.5GB".

    Returns
    -------
    size_in_bytes: int
        The file size in bytes.

    Examples
    --------
    >>> parse_file_size("10KB")
    10240

    >>> parse_file_size("2.5GB")
    2684354560

    """

    # Reference: https://stackoverflow.com/a/60708339/9707202
    # Author: https://stackoverflow.com/users/2002471/chicks

    units = {"B": 1,
             "KB": 2**10,
             "MB": 2**20,
             "GB": 2**30,
             "TB": 2**40}

    size = size.upper()

    if not re.match(r' ', size):
        size = re.sub(r'([KMGT]?B)', r' \1', size)

    number, unit = [string.strip() for string in size.split()]

    return int(float(number) * units[unit])


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


def length_to_slices_of_indexes(length: int, step: int) -> Iterator[slice]:
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
                numel: int,
                preferred_axes: Optional[list[Union[int, list]]] = None
                ) -> tuple[int, ...]:
    """
    Calculate how to best split a np.ndarray in smaller chunks.

    Parameters
    ----------
    shape : tuple
        The shape of a np.ndarray, e.g.: `arr.shape`
    numel : int
        Maximum number of elements in the resulting chunk, e.g.:
        `chunk[0] * chunk[1] * ... chunk[N] <= numel`
    preferred_axes : list, optional
        A list of preferred axes for chunking, by default None. Axes are
        treated in order of preference unless inside a sublist. In this case,
        they are treated with equal preferency, e.g.:

        [0, 1] -> allocate maximum possible to axis 0, then allocate whats is
        left to axis 1, then allocate what is left "equally" to remaining axes.

        [[2, 3], 1, 0] -> allocate "equally" to axes 2 and 3, then allocate
        what is left to axis 1, then allocate what is left to axis 0, then
        allocate what is left "equally" to remaining axes.

    Returns
    -------
    chunk : tuple
        Chunk shape.

    Notes
    -----
    This function attempts to find a "balanced" solution, not simply the
    "best" one, e.g.: for a array with shape (1000, 30, 200, 100) and a goal
    of 1024 elements, a "balanced" solution is (6, 6, 5, 5) -> 900, while
    the "best" solution would be an unbalanced (1024, 1, 1, 1) -> 1024.

    Examples
    --------
    >>> shape2chunk(shape=(400, 30, 200, 100), numel=1024)
    (6, 6, 5, 5)

    >>> shape2chunk(shape=(400, 30, 200, 100), numel=1024,
                    preferred_axes=[0, 1])
    (400, 2, 1, 1)

    >>> shape2chunk(shape=(400, 30, 200, 100), numel=1024,
                    preferred_axes=[[0, 1]])
    (34, 30, 1, 1)

    >>> shape2chunk(shape=(400, 30, 20, 10), numel=1024,
                    preferred_axes=[[2, 3], 1, 0])
    (1, 5, 20, 10)

    >>> shape2chunk(shape=(400, 30, 20, 10), numel=1024,
                    preferred_axes=[[2, 3], [1, 0]])
    (2, 2, 20, 10)

    """

    _ = [int(str(size)) for size in shape]
    if 0 in shape:
        raise ValueError("No axis can have size 0")

    _ = int(str(numel))
    if not numel:
        raise ValueError("Number of elements can't be zero")

    preferred_axes = [] if preferred_axes is None else preferred_axes

    rngs = [range(1, size) for size in shape]

    for axes in preferred_axes:

        axes = axes if isinstance(axes, Iterable) else [axes]

        rngs2 = [rng
                 if axis in axes
                 else range(rng.start, rng.start)
                 for axis, rng in enumerate(rngs)]

        chunks = _ranges2product(rngs2, numel)

        rngs = [range(chunks[axis], chunks[axis])
                if axis in axes
                else rng
                for axis, rng in enumerate(rngs)]

    return _ranges2product(rngs, numel)


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

    Notes
    -----
    This function attempts to find a "balanced" solution, not simply the
    "best" one, e.g.: for an array with shape (1000, 30, 200, 100) and a goal
    of 1024 elements, a "balanced" solution is (6, 6, 5, 5) -> 900, while
    the "best" solution would be an unbalanced (1024, 1, 1, 1) -> 1024.

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

    arr = (np.ones((len(stops), max_size))
           .cumsum(axis=1)
           .astype(int)
           .T
           .clip(min=starts, max=stops))

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
def any2datetime(dt: Union[datetime.datetime,
                           datetime.date,
                           np.datetime64,
                           str],
                 dt_fmt: Optional[str] = None
                 ) -> datetime.datetime:
    """
    Convert `dt` to datetime.datetime object.

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

    >>> any2datetime("198312190403")   # assumes yyyymmdd
    datetime.datetime(1983, 12, 19, 4, 3)

    >>> any2datetime("01/02/2010")   # assumes dd/mm/yyyy
    datetime.datetime(2010, 2, 1, 0, 0)

    >>> any2datetime("d=19 m=12 a=1983")
    ValueError: time data 'd=19 m=12 a=1983' does not match ...

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

    # using an specific list of formats to avoid "month-day-year" problems
    # with pandas and dateutil parsers
    dt_fmts = (
        list(accumulate(["%Y", "%m", "%d", "%H", "%M", "%S", ".%f"]))
        + list(accumulate(["%Y", "%m", "%d", "T%H", "%M", "%S", ".%f"]))
        + list(accumulate(["%Y", "-%m", "-%d", " %H", ":%M", ":%S", ".%f"]))
        + list(accumulate(["%Y", "-%m", "-%d", "T%H", ":%M", ":%S", ".%f"]))
        + list(accumulate(["%d/%m/%Y", " %H", ":%M", ":%S", ".%f"]))
    )

    for dt_fmt in dt_fmts:
        try:

            # convert to datetime obj
            dt_dt = datetime.datetime.strptime(dt, dt_fmt)

            # go back to str to make sure the conversion is correct
            # Note: this is necessary because strptime allows numbers with
            # single digits, e.g.:
            # strptime('19831912', '%Y%m%d%H%M') -> (1983, 1, 9, 1, 2)
            dt_str = datetime.datetime.strftime(dt_dt, dt_fmt)

            assert dt == dt_str

            return dt_dt

        except Exception as e:
            pass

    raise ValueError(f"time data '{dt}' does not match any of these formats: "
                     + ", ".join(dt_fmts))
