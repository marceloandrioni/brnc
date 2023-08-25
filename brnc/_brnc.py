#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["HOW_TO_USE_IT"]


from typing import Optional, Union
import time
import numpy as np
import pandas as pd
import xarray as xr

from ._axis import da2axis
from ._common import (index_of_valid_value_along_axis, valid_value_along_axis,
                      number2int, shape2chunk)
from ._types import INT_FLOAT_ANY2DT

import logging

# use GMT/Zulu time
logging.Formatter.converter = time.gmtime
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%SZ",
                    force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


HOW_TO_USE_IT = ("Importing this module automatically adds a 'br' accessor to "
                 "xarray DataArray and Dataset, e.g.: da.br and ds.br")


class DaDsMixin:
    """
    Methods common to DataArrays and Datasets.
    """

    def __init__(self, xarray_obj) -> None:
        self._obj = xarray_obj

    @property
    def dx(self) -> Union[xr.DataArray, xr.Dataset]:
        return self._obj

    def sel_nearest(self,
                    keep_as_dim: bool = False,
                    **kwargs: INT_FLOAT_ANY2DT
                    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Select the nearest data point to the specified value along the dimension.

        Parameters
        ----------
        keep_as_dim : bool, optional
            Flag indicating whether to keep the dimension for the selected
            value as a separate dimension.
        **kwargs : int, float, datetime.datetime, datetime.date, np.datetime64,
                   str
            Keyword arguments representing the dimension and corresponding
            value.

        Returns
        -------
        dx : xr.DataArray or xr.Dataset
            DataArray/Dataset with the nearest data point. Selected dimensions
            will be returned with size 1 if keep_as_dim`is True, else, the
            dimension will be dropped.

        Examples
        --------
        >>> ds = xr.tutorial.load_dataset("air_temperature_gradient")
        >>> ds.br.sel_nearest(time="2013-01-02 01:00", lat=51, lon=246)

        """

        def f(x):
            return [x] if keep_as_dim else x

        isel_kwargs = {dim: f(da2axis(self.dx[dim]).find_index(value))
                       for dim, value in kwargs.items()}

        return self.dx.isel(**isel_kwargs)

    def sel_around(self,
                   **kwargs: INT_FLOAT_ANY2DT
                   ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Select the two data points around the specified value along the
        dimension.

        Parameters
        ----------
        **kwargs : int, float, datetime.datetime, datetime.date, np.datetime64,
                   str
            Keyword arguments representing the dimension and corresponding
            values.

        Returns
        -------
        dx : xr.DataArray or xr.Dataset
            DataArray/Dataset with selected data points. Selected dimensions
            will be returned with size 2.

        Examples
        --------
        >>> ds = xr.tutorial.load_dataset("air_temperature_gradient")
        >>> ds.br.sel_around(time="2013-01-02 01:00", lat=51, lon=246)

        """

        isel_kwargs = {dim: da2axis(self.dx[dim]).find_indexes(value)
                       for dim, value in kwargs.items()}

        return self.dx.isel(**isel_kwargs)

    def sel_slice(self,
                  force_inclusive: bool = False,
                  **kwargs: slice
                  ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Select a slice along the specified dimension.

        Parameters
        ----------
        force_inclusive : bool, optional
            Flag indicating whether the slice should be expanded to forcefully
            include the values at the start and end of the slice.
        **kwargs : slice
            Keyword arguments representing the dimension and corresponding
            slice.

        Returns
        -------
        dx : xr.DataArray or xr.Dataset
            Sliced DataArray/Dataset

        Examples
        --------
        >>> ds = xr.tutorial.load_dataset("air_temperature_gradient")

        >>> ds.br.sel_slice(time=slice("2013-01-02 01:00",
                                       "2013-04-03 03:00"),
                            lat=slice(51, 59),
                            lon=slice(246, 254))

        >>> ds.br.sel_slice(time=slice("2013-01-02 01:00",
                                       "2013-04-03 03:00"),
                            lat=slice(51, 59),
                            lon=slice(246, 254),
                            force_inclusive=True)

        """

        isel_kwargs = dict()
        for dim, slc in kwargs.items():
            axis = da2axis(self.dx[dim])
            isel_kwargs[dim] = axis.find_indexes_between(slc.start,
                                                         slc.stop,
                                                         force_inclusive)

        return self.dx.isel(**isel_kwargs)


@xr.register_dataarray_accessor("br")
class BrDA(DaDsMixin):

    @property
    def da(self) -> xr.DataArray:
        return self._obj

    @property
    def name(self) -> str:
        return getattr(self.da, "name", "unnamed")

    def err(self, msg: str) -> None:
        """Raise a ValueError with message msg."""
        raise ValueError(f"DataArray '{self.name}': {msg}")

    def info(self, msg: str) -> None:
        """Info message msg."""
        log.info(f"DataArray '{self.name}': {msg}")

    def warn(self, msg: str) -> None:
        """Warning message msg."""
        log.warning(f"DataArray '{self.name}': {msg}")

    def load(self, **kwargs: int) -> xr.DataArray:

        # just return it if it was already loaded
        if self.da._in_memory:
            return self.da

        self.info("loading data in memory")

        # load the whole DataArray if no kwargs
        if len(kwargs) == 0:
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

    def chunk(self,
              preferred_dims: Optional[list[str]] = None,
              preferred_in_order: bool = False,
              size: int = 4096
              ) -> xr.DataArray:
        """
        Set the chunks.

        This is useful to improve IO when writing/reading a NetCDF file. The
        chunks tell how the N dimension data matrix will be split in smaller
        blocks/chunks to improve IO.

        Parameters
        ----------
        preferred_dims : list, optional
            List of preferred dimensions to use for chunking. This is
            useful if the user know in advance that a DataArray will be most
            accessed to retrieve time-series/time-profile-series for a location
            or to retrieve spatial horizontal sections (e.g. to plot a map),
            e.g.: assuming a 4D (time, depth, lat, lon) DataArray

            * if the user wants to retrieve time series with greater efficiency,
            preferred_dims=['time'] should be used

            * for time-profile-series, preferred_dims=['time', 'depth']

            * for horizontal (lat x lon) sections,
            preferred_dims=['latitude', 'longitude']

            This will increase the chunkshape in the chosen dimension (while
            reducing in the others) to reduce the number of necessary disk
            access calls. If None, treat all dimensions equally.
            Default: None
        preferred_in_order : bool, optional
            Flag indicating whether the preferred dimensions should be chunked
            in order. If there is more than one dimension in `preferred_dims`
            and preferred_in_order is True, optimize as much as possible the
            first dimension given, then the second dimension and so on. If
            preferred_in_order is False, all the dimensions in preferred_dims
            are optimized equally.
        size : int, optional
            Maximum size in bytes of individual chunks. Best results are
            obtained using a size equal to a disk block size (4096B = 4KB).
            Default: 4096

        Returns
        -------
        chunked : xarray.DataArray
            Chunked DataArray.
            `chunkshape[0] * chunkshape[1] * ... chunkshape[N] * itemsize`
            should be less or equall to `size`.

        Notes
        -----
        This function attempts to find a "balanced" solution, not simply the
        "best" one, e.g.: for an array with shape (1000, 30, 200, 100) and a goal
        of 1024 elements, a "balanced" solution is (6, 6, 5, 5) -> 900, while
        the "best" solution would be an unbalanced (1024, 1, 1, 1) -> 1024.

        Examples
        --------
        >>> ds = xr.tutorial.load_dataset("eraint_uvz")
        >>> da = ds["u"].br.chunk(preferred_dims=["latitude", "longitude"])
        >>> da.encoding["chunksizes"]
        (1, 1, 45, 45)

        """

        # references:
        # https://www.unidata.ucar.edu/blogs/developer/en/entry/chunking_data_why_it_matters
        # https://www.unidata.ucar.edu/blogs/developer/en/entry/chunking_data_choosing_shapes
        # https://www.unidata.ucar.edu/software/netcdf/workshops/most-recent/nc4chunking/index.html
        # https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_perf_chunking.html
        # https://www.unidata.ucar.edu/blog_content/data/2013/chunk_shape_3D.py
        # https://gist.github.com/feliperiosg/21872715b2de8300f0cb52086f363d6b
        # http://wiki.seas.harvard.edu/geos-chem/index.php/Working_with_netCDF_data_files#Chunking_and_deflating_a_netCDF_file_to_improve_I.2FO
        #
        # following
        # https://gist.github.com/feliperiosg/21872715b2de8300f0cb52086f363d6b
        # 'good shape' for chunks means that the number of chunks accessed to read
        # either kind of 1D or 2D subset is approximately equal, and the size of
        # each chunk (uncompressed) is no more than chunkSize, which is often a disk
        # block size.
        #
        # To be sure of disk block size
        # sudo blockdev --getbsz /dev/sdX
        # but it is probably 4096 bytes = 4 KB
        #
        # The block size is not a "hard" threshold, it is just a recommendation.
        # It is possible to user greater or smaller values (e.g. use
        # [1, lat_fullsize, lon_fullsize] to retrieve the whole "map")
        #
        # For a dataset with hourly single level data on a lat/177 x lon/173 grid,
        # the retrieval time for one year (~8760 times) of monthly aggregated files
        # in THREDDS was:
        #   simultaneous retrieval of 8 variables:
        #     50 seconds using _ChunkSizes = 1, 177, 173
        #     11 seconds using _ChunkSizes = 744, 2, 1   (4.54x faster)
        #   simultaneous retrieval of 2 variables:
        #     12 seconds using _ChunkSizes = 1, 177, 173
        #     2.7 seconds using _ChunkSizes = 744, 2, 1   (4.44x faster)
        #
        # For a dataset of 12 monthly files, each with dimensions:
        # time = ~242 ;
        # depth = 40 ;
        # latitude = 126 ;
        # longitude = 175 ;
        # aggregated in THREDDS and the simultaneous retrieval of 2 variables (U and V):
        # 6.3 to 6.5s   retrieving 3d current profile in files optimized with chuncking along tz (78, 13, 1, 1)
        # 3.8 to 4.2s   retrieving 2d surface current in files optimized with chuncking along tz
        # 7.8 to 8.2s   retrieving 3d current profile in files optimized with chuncking along t (242, 1, 2, 2)
        # 0.9 to 1.5s   retrieving 2d surface current in files optimized with chuncking along t
        # 14.2 to 14.7s retrieving 3d current profile in files optimized with chuncking along yx (1, 1, 27, 37)
        # 5.1 to 5.8s   retrieving 2d surface current in files optimized with chuncking along yx
        #
        # If the 3d dataset will be used primarily to retrieve only the surface
        # current, it may be worth it to optimize only along the time dimension
        # (t=1.5s vs tz=4.2s), considering the retrieval is almost 3x faster and
        # the full 3d retrieval is only 25% slower (t=8.2s vs tz=6.5s).
        #
        # Note: this probably could be done in a more elegant way using a
        # scipy.optimize.minimize (or similar), but the problem is that the
        # chunks must be integers.

        # get size (in bytes) of individual values. Get dtype from
        # da.encoding["dtype"] if available, else, use da.dtype. Prefer
        # encoding because this is the dtype that will be used when saving the
        # data to disk.
        itemsize = self.da.encoding.get("dtype",
                                        self.da.dtype).itemsize

        if not itemsize:
            self.err("itemsize can't be zero or None")

        if 0 in self.da.sizes.values():
            self.err("no dimension can be less than 1")

        numel = number2int(size / itemsize)
        if numel < 1:
            self.err("size divided by itemsize can't be less than 1")

        preferred_axes = None
        if preferred_dims:
            preferred_axes = [self.da.get_axis_num(dim)
                              for dim in preferred_dims]

        chunks = dict(zip(
            self.da.dims,
            shape2chunk(shape=self.da.shape,
                        numel=numel,
                        preferred_axes=preferred_axes,
                        preferred_in_order=preferred_in_order)))

        self.info(
            f"setting chunks '{chunks}' with {np.r_[chunks].prod() * itemsize} "
            f"bytes ({np.r_[chunks].prod()} x {itemsize} bytes)")

        # keep the original da as is
        da = self.da.copy()

        # need to remove so that chunkshape is really applied
        for x in ["original_shape", "_ChunkSize", "_ChunkSizes"]:
            _ = da.encoding.pop(x, None)

        # setting the chunksize directly instead of using da.chunk
        da.encoding["chunksizes"] = tuple(chunks.values())

        return da


@xr.register_dataset_accessor('br')
class BrDS(DaDsMixin):

    @property
    def ds(self) -> xr.DataArray:
        return self._obj

    def err(self, msg: str) -> None:
        """Raise a ValueError with message msg."""
        raise ValueError(f"Dataset: {msg}")

    def info(self, msg: str) -> None:
        """Info message msg."""
        log.info(f"Dataset: {msg}")

    def warn(self, msg: str) -> None:
        """Warning message msg."""
        log.warning(f"Dataset: {msg}")
