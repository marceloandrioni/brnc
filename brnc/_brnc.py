#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

__all__ = ["index_of_valid_value_along_axis",
           "valid_value_along_axis",
           "BrDA"]

# %reset -f

from typing import Optional
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


def index_of_valid_value_along_axis(arr: np.ndarray,
                                    *,
                                    axis: int,
                                    position: str = 'first',
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
        along the axis. Default is 'first'.

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

    >>> index_of_valid_value_along_axis(arr, axis=0, position='first')
    array(1, 0, 0)

    >>> index_of_valid_value_along_axis(arr, axis=1, position='last')
    array([ 2,  2,  1, -1])

    """

    ones = np.ones(arr.shape)

    valid = np.where(~np.isnan(arr), ones, np.nan)

    valid_indexes = ((ones.cumsum(axis=axis) - 1) * valid)

    positions = {'first': np.nanmin, 'last': np.nanmax}

    indexes = positions[position](valid_indexes, axis=axis)

    # use -1 if all elements along axis are nan
    # Note: -1 is a valid index when searching the array, so be careful
    indexes = np.where(np.isnan(indexes), -1, indexes).astype(int)

    return indexes


def valid_value_along_axis(arr: np.ndarray,
                           *,
                           axis: int,
                           position: str = 'first',
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
        Default is 'first'.

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

    >>> valid_value_along_axis(arr, axis=0, position='first')
    array([7., 2., 5.])

    >>> valid_value_along_axis(arr, axis=1, position='last')
    array([ 5.,  8.,  4., nan])
    """

    indexes = index_of_valid_value_along_axis(arr,
                                              axis=axis,
                                              position=position)

    sizes = indexes.shape

    other_axes = np.ogrid[tuple([slice(size) for size in sizes])]
    # other_axes = np.meshgrid(*[range(size) for size in sizes],
    #                          sparse=True,
    #                          indexing='ij')

    other_axes.insert(axis, indexes)

    return np.where(indexes < 0, np.nan, arr[tuple(other_axes)])


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
        """Check whether the DataArray is of a numeric dtype.

        Returns
        -------
        boolean
            Whether or not the array or dtype is of a numeric dtype.

        """
        return pd.api.types.is_numeric_dtype(self.da)

    def index_of_valid_value_along_dimension(self,
                                             dim: Optional[str] = None,
                                             axis: Optional[int] = None,
                                             position: str = 'first',
                                             ) -> xr.DataArray:
        """
        Get the index of the first or last valid value along a specific
        dimension or axis.

        Parameters
        ----------
        dim : str, optional
            The name of the dimension along which to get the index.
            Either `dim` or `axis` must be provided.
        axis : int, optional
            The axis along which to get the index.
            Either `dim` or `axis` must be provided.
        position : str, optional
            Specifies whether to retrieve the index of the first or last valid
            value along the dimension or axis.
            Default is 'first'.

        Returns
        -------
        xr.DataArray
            xr.DataArray containing the index of the first or last valid value
            along the specified dimension or axis.

        """

        if not any([dim, axis]):
            self.err('dim or axis argument must be supplied')

        # Note: if da is a <class 'dask.array.core.Array'>, the code fails with:
        # "NotImplementedError: Don't yet support nd fancy indexing"
        # so call load to make sure da is a np.array with data in memory.

        return self.da.br.load().reduce(
            index_of_valid_value_along_axis,
            dim=dim,
            axis=axis,
            keep_attrs=True,
            position=position)

    def valid_value_along_dimension(self,
                                    dim: Optional[str] = None,
                                    axis: Optional[int] = None,
                                    position: str = 'first',
                                    ) -> xr.DataArray:
        """
        Get the first or last valid value along a specific dimension or axis.

        Parameters
        ----------
        dim : str, optional
            The name of the dimension along which to get the value.
            Either `dim` or `axis` must be provided.
        axis : int, optional
            The axis along which to get the value.
            Either `dim` or `axis` must be provided.
        position : str, optional
            Specifies whether to retrieve the first or last valid value along
            the dimension or axis.
            Default is 'first'.

        Returns
        -------
        xr.DataArray
            xr.DataArray containing the first or last valid value along the
            specified dimension or axis.

        """

        if not any([dim, axis]):
            self.err('dim or axis argument must be supplied')

        # Note: if da is a <class 'dask.array.core.Array'>, the code fails with:
        # "NotImplementedError: Don't yet support nd fancy indexing"
        # so call load to make sure da is a np.array with data in memory.

        return self.da.br.load().reduce(
            valid_value_along_axis,
            dim=dim,
            axis=axis,
            keep_attrs=True,
            position=position)

