#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["HOW_TO_USE_IT"]


from typing import Optional, Union
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr

from ._axis import AxisFactory
from ._common import (index_of_valid_value_along_axis, valid_value_along_axis,
                      number2int, shape2chunk, length_to_slices_of_indexes,
                      dict_prod, humanize_file_size, dehumanize_file_size)
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

    @property
    def dims(self):
        """
        Mapping from dimension names to Axis (AxisFloat, AxisInt or AxisTime).
        """

        axfac = AxisFactory()

        return {dim: axfac.from_dataarray(self.dx[dim])
                for dim in self.dx.dims}

    def sel_nearest(self,
                    keep_as_dim: bool = False,
                    **dims_kws: INT_FLOAT_ANY2DT,
                    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Select the nearest data point to the specified value along the dimension.

        Parameters
        ----------
        **dims_kws : int, float, datetime.datetime, datetime.date, np.datetime64,
                     str
            Dimension and corresponding value.
        keep_as_dim : bool, optional
            Flag indicating whether to keep the dimension for the selected
            value as a separate dimension.

        Returns
        -------
        dx : xr.DataArray or xr.Dataset
            DataArray/Dataset with the nearest data point. Selected dimensions
            will be returned with size 1 if `keep_as_dim` is True, else, the
            dimension will be dropped.

        Examples
        --------
        >>> ds = xr.tutorial.load_dataset("air_temperature_gradient")
        >>> ds.br.sel_nearest(time="2013-01-02 01:00", lat=51, lon=246)

        """

        def f(x):
            return [x] if keep_as_dim else x

        dims = self.dims

        isel_kwargs = {dim: dims[dim].find_index_nearest(value)
                       for dim, value in dims_kws.items()}

        return self.dx.isel(**isel_kwargs)

    def sel_around(self,
                   **dims_kws: INT_FLOAT_ANY2DT
                   ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Select the two data points around the specified value along the
        dimension.

        Parameters
        ----------
        **dims_kws : int, float, datetime.datetime, datetime.date, np.datetime64,
                     str
            Dimension and corresponding value.

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

        dims = self.dims

        isel_kwargs = {dim: dims[dim].find_indexes_around(value)
                       for dim, value in dims_kws.items()}

        return self.dx.isel(**isel_kwargs)

    def sel_slice(self,
                  force_inclusive: bool = False,
                  **dims_kws: slice,
                  ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Select a slice along the specified dimension.

        Parameters
        ----------
        **dims_kws : slice
            Dimension and corresponding slice.
        force_inclusive : bool, optional
            Flag indicating whether the slice should be expanded to forcefully
            include the values at the start and end of the slice.

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

        dims = self.dims

        isel_kwargs = {dim: dims[dim].find_indexes_between(slc.start,
                                                           slc.stop,
                                                           force_inclusive)
                       for dim, slc in dims_kws.items()}

        return self.dx.isel(**isel_kwargs)


@xr.register_dataarray_accessor("br")
class BrDA(DaDsMixin):

    @property
    def da(self) -> xr.DataArray:
        return self._obj

    @property
    def name(self) -> str:

        names = [self.da.name,
                 "unnamed"]

        return next(filter(None, names))

    def err(self, msg: str) -> None:
        """Raise a ValueError with message msg."""
        raise ValueError(f"DataArray '{self.name}': {msg}")

    def info(self, msg: str) -> None:
        """Info message msg."""
        log.info(f"DataArray '{self.name}': {msg}")

    def warn(self, msg: str) -> None:
        """Warning message msg."""
        log.warning(f"DataArray '{self.name}': {msg}")

    def load(self) -> xr.DataArray:

        # just return it if it was already loaded
        if self.da._in_memory:
            return self.da

        self.info("loading data in memory: "
                  f"{humanize_file_size(self.da.nbytes)}")

        # load() returns the data and changes the obj itself
        # compute() returns the data and keeps the obj unchanged
        #
        # using compute() to avoid using too much memory in a ds loop, e.g:
        # assuming 1GB per varible, the first iteration using load() would need
        # 1GB, the second 2GB and the third 3GB. Using compute(), only 1GB would
        # be needed.
        #
        # print("func", "var", "ds[var]._in_memory", "da2._in_memory")
        # for func in ["compute", "load"]:
        #     with xr.tutorial.open_dataset("air_temperature_gradient") as ds:
        #         # xr.Dataset().to_netcdf(file)
        #         for var in list(ds.data_vars):
        #             da2 = getattr(ds[var], func)()
        #             # da2.to_netcdf(file, mode="a")
        #             print(func, var, ds[var]._in_memory, da2._in_memory)
        #
        # func var ds[var]._in_memory da2._in_memory
        # compute Tair False True
        # compute dTdx False True
        # compute dTdy False True
        # load Tair True True
        # load dTdx True True
        # load dTdy True True

        return self.da.compute()

    def _dims_steps_to_dims_slices(self, dims_steps: dict) -> dict:

        return {dim: length_to_slices_of_indexes(self.da[dim].size, step)
                for dim, step in dims_steps.items()}

    def _tqdm_description(self, pbar, d: dict) -> str:

        slices_repr = ", ".join(
            [f"{dim}:[{slc.start}:{slc.stop})/{self.da[dim].size}"
             for dim, slc in d.items()])

        # tqdm uses {percentage:3.0f}% to represent percentage. This may
        # incorrectly display 100% in the last few iterations, e.g.:
        # f"{99.5:3.0f}" -> 100
        # https://github.com/tqdm/tqdm/issues/1398
        perc = np.floor(100 * pbar.n / pbar.total)

        return f"{slices_repr}: {perc:.0f}%"

    def load_by_step(self, **dims_kws: int) -> xr.DataArray:

        # just return it if it was already loaded
        if self.da._in_memory:
            return self.da

        # load the whole DataArray if no dims_kws
        if len(dims_kws) == 0:
            return self.load()

        self.info("loading data in memory: "
                  f"{humanize_file_size(self.da.nbytes)}")

        # split the DataArray in chunks, load each chunk individually and merge
        slices = dict_prod(self._dims_steps_to_dims_slices(dims_kws))

        bar_format = "{desc}|{bar}| {n_fmt}/{total_fmt} [{elapsed}]"

        with tqdm(slices,
                  total=len(slices),
                  bar_format=bar_format) as pbar:

            das = []
            for d in slices:
                pbar.set_description(self._tqdm_description(pbar, d))
                das.append(self.da.isel(**d).compute())
                # time.sleep(1)
                pbar.update(1)

            pbar.set_description(self._tqdm_description(pbar, d))

            da = xr.combine_by_coords(das, combine_attrs="identical")

        # xr.combine_by_coords returns a Dataset if da.name is not None
        if isinstance(da, xr.Dataset):
            da = da[list(da.data_vars)[0]]

        return da

    def load_by_size(self,
                     pref_dims: Optional[list[Union[str, list]]] = None,
                     size: Union[int, str] = "20MB"
                     ) -> xr.DataArray:

        itemsize = self.da.dtype.itemsize

        if not itemsize:
            self.err("itemsize can't be zero or None")

        if 0 in self.da.sizes.values():
            self.err("no dimension can be less than 1")

        size = dehumanize_file_size(size) if isinstance(size, str) else size

        if self.da.nbytes <= size:
            return self.load()

        numel = number2int(size / itemsize)
        if numel < 1:
            self.err("size divided by itemsize can't be less than 1")

        pref_axes = None
        if pref_dims:
            pref_axes = [self.da.get_axis_num(dim)
                         for dim in pref_dims]

        chunks = dict(zip(
            self.da.dims,
            shape2chunk(shape=self.da.shape,
                        numel=numel,
                        pref_axes=pref_axes)))

        return self.load_by_step(**chunks)

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

        return self.load().reduce(
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

        return self.load().reduce(
            valid_value_along_axis,
            dim=dim,
            keep_attrs=True,
            position=position)

    def constant_validity_along_dimension(self,
                                          dim: str
                                          ) -> xr.DataArray:

        raise NotImplementedError

        # True if the value behaviour is persistent along dimension, e.g.:
        # if all values along dim are nan/inf -> True
        # if all values along dim are valid -> True
        # if there is nan/inf and valid values along dim -> False
        #
        # Useful to check if there is no missing data along time dimension in
        # a 3D (time,lat,lon) or 4D (time, depth, lat, lon) variable

        # def f(arr):
        #     # np.isfinite checks for nan and inf
        #     return np.all(np.isfinite(arr)) or np.all(~np.isfinite(arr))
        #     return np.all(np.isnan(arr)) or np.all(~np.isnan(arr))
        # should I use np.isnan or np.isfinite? nan is to be expected, but inf is not

        # arr = np.random.rand(20, 10, 5)
        # arr[1,1,1] = np.nan

        # def func1(arr, axis):

        #     def f(arr):

        #         # np.isfinite checks for nan and inf
        #         out = np.isfinite(arr)
        #         return np.all(out) or np.all(~out)

        #     return np.apply_along_axis(f,
        #                                axis,
        #                                arr)

        # def func2(arr, axis):
        #     # return true if values along axis are all valid or all invalid
        #     # return false if there is valid and invalid valus along axis
        # #     aux = np.isnan(arr).sum(axis=axis)
        #     aux = np.isfinite(arr).sum(axis=axis)
        #     return (aux == 0) | (aux == arr.shape[axis])

        # def func(da):
        #     return da.reduce(func1,
        #                      dim="time")
        #
        # func2 if much (300x) faster than func1


    # mark only the border points

    # infile = "/tmp/cmems_global-analysis-forecast-phy-001-024.nc"

    # ds = xr.open_dataset(infile)

    # ds["mask"] = (("latitude", "longitude"),
    #               np.where(np.isnan(ds["uo"].isel(time=0, depth=0).values), 0, 1))

    # mask = ds["mask"].values

    # mlon, mlat = np.meshgrid(ds["longitude"], ds["latitude"])


    # def in_border(mask: np.ndarray) -> np.ndarray:

    #     border = np.full(mask.shape, False, dtype=bool)
    #     for index, value in np.ndenumerate(mask):

    #         if not value:
    #             continue

    #         # True if point is in grid border
    #         for axis in range(mask.ndim):

    #             if border[index]:
    #                 continue

    #             if index[axis] == 0 or index[axis] == mask.shape[axis] - 1:
    #                 border[index] = True

    #         # True if any of the surrounding points is False
    #         slices = tuple([slice(index[axis] - 1, index[axis] + 2)
    #                         for axis in range(mask.ndim)])

    #         if mask[slices].sum() != mask[slices].size:
    #             border[index] = True
    #             continue

    #     return border

    # border = in_border(mask)

    # df = pd.DataFrame(data=np.column_stack((mlon.flatten(),
    #                                         mlat.flatten(),
    #                                         border.flatten())),
    #                   columns=["longitude", "latitude", "border"])
    # df["border"] = df["border"].astype(bool)
    # df = df[df["border"]][["longitude", "latitude"]]


    # df = ds["mask"].to_dataframe().reset_index()
    # df = df[df["mask"].astype(bool)][["longitude", "latitude"]]

    # fig, ax = plt.subplots()
    # ax.pcolormesh(mlon, mlat, mask, edgecolors='k', alpha=0.3)
    # ax.plot(df["longitude"], df["latitude"], ".r")

    # convex = shapely.MultiPoint(df.values).convex_hull
    # # ax.plot(*convex.boundary.xy)

    # concave = shapely.concave_hull(shapely.MultiPoint(df.values),
    #                                ratio=0.001,
    #                                allow_holes=False)
    # ax.plot(*concave.boundary.xy, 'g-')
    # ax.axis('equal')

    def chunk(self,
              pref_dims: Optional[list[Union[str, list]]] = None,
              size: Union[int, str] = 4096
              ) -> xr.DataArray:
        """
        Set the chunks.

        This is useful to improve IO when writing/reading a NetCDF file. The
        chunks tell how the N dimension data matrix will be split in smaller
        blocks/chunks to improve IO.

        Parameters
        ----------
        pref_dims : list, optional
            A list of preferential dimensions for chunking, by default None.
            Dimensions are treated in order of preference unless inside a
            sublist. In this case, they are treated with equal preferency, e.g.:

            ["time"] -> allocate maximum possible to dimension time, then
            allocate what is left "equally" to remaining axes.

            [["longitude", "latitude"], "depth"] -> allocate "equally" to
            dimensions "longitude" and "latitude", then allocate what is left
            to dimension "depth", then allocate what is left "equally" to
            remaining dimensions.

            This is useful if the user know in advance that a DataArray will be
            most accessed to retrieve time-series/time-profile-series for a
            location or to retrieve spatial horizontal sections (e.g. to plot a
            map), e.g.: assuming a 4D (time, depth, lat, lon) DataArray

            * if the user wants to retrieve time series with greater efficiency,
            pref_dims=["time"] should be used

            * for time-profile-series, pref_dims=[["time", "depth"]]

            * for horizontal (lat x lon) sections,
            pref_dims=[["latitude", "longitude"]]

            This will increase the chunkshape in the chosen dimension (while
            reducing in the others) to reduce the number of necessary disk
            access calls. If None, treat all dimensions equally.
            Default: None
        size : int or str, optional
            Maximum size of individual chunks. Best results are obtained using
            a size equal to a disk block size (4096B = 4KB), but this is not a
            "hard" threshold.
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
        >>> da = xr.tutorial.load_dataset("air_temperature")["air"]
        >>> da.br.chunk(pref_dims=["time"]).encoding["chunksizes"]
        (2048, 1, 1)
        >>> da.br.chunk(pref_dims=[["lat", "lon"]]).encoding["chunksizes"]
        (1, 25, 53)

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

        size = dehumanize_file_size(size) if isinstance(size, str) else size

        numel = number2int(size / itemsize)
        if numel < 1:
            self.err("size divided by itemsize can't be less than 1")

        pref_axes = None
        if pref_dims:
            pref_axes = [self.da.get_axis_num(dim)
                         for dim in pref_dims]

        chunks = dict(zip(
            self.da.dims,
            shape2chunk(shape=self.da.shape,
                        numel=numel,
                        pref_axes=pref_axes)))

        self.info(
            f"setting chunks '{chunks}' with "
            f"{np.r_[list(chunks.values())].prod() * itemsize} bytes "
            f"({np.r_[list(chunks.values())].prod()} elements x {itemsize} bytes)")

        # keep the original DataArray as is
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

    @property
    def name(self) -> str:

        names = [self.ds.encoding.get("source"),
                 "unnamed"]

        return next(filter(None, names))

    def err(self, msg: str) -> None:
        """Raise a ValueError with message msg."""
        raise ValueError(f"Dataset '{self.name}': {msg}")

    def info(self, msg: str) -> None:
        """Info message msg."""
        log.info(f"Dataset '{self.name}': {msg}")

    def warn(self, msg: str) -> None:
        """Warning message msg."""
        log.warning(f"Dataset '{self.name}': {msg}")
