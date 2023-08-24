#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
import numpy as np
import xarray as xr
import pyinterp   # type: ignore

from ._types import INT_FLOAT_DT64, ANY2DT
from ._common import any2datetime


class AxisMixin:

    def _raise_if_outside_range(self, value: INT_FLOAT_DT64) -> None:
        """Raise an Exception if value is outside the [min, max] range."""

        # Note: pyinterp find_index returns valid (>= 0) index for values
        # in the range [min_value - step/2, max_value + step/2]

        if self.min_value() <= value <= self.max_value():
            return

        raise ValueError(f"Requested value {value} is outside the axis range "
                         f"[{self.min_value()}, {self.max_value()}]")

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self)

    @property
    def minmax(self) -> list[INT_FLOAT_DT64]:
        return [self.min_value(), self.max_value()]

    def find_index(self, value: INT_FLOAT_DT64) -> int:
        self._raise_if_outside_range(value)
        return super().find_index(np.r_[value]).item()

    def find_index_le(self, value: INT_FLOAT_DT64) -> int:

        idx = self.find_index(value)

        if self.values[idx] <= value:
            return idx

        return idx - 1 if self.is_ascending() else idx + 1

    def find_index_ge(self, value: INT_FLOAT_DT64) -> int:

        idx = self.find_index(value)

        if self.values[idx] >= value:
            return idx

        return idx + 1 if self.is_ascending() else idx - 1

    def find_value_nearest(self, value: INT_FLOAT_DT64) -> INT_FLOAT_DT64:
        return self.values[self.find_index(value)]

    def find_indexes(self, value: INT_FLOAT_DT64) -> list[int]:

        self._raise_if_outside_range(value)

        if len(self.values) == 1:
            raise ValueError(
                "Can't get indexes around value for an axis with length 1")

        return list(super().find_indexes(np.r_[value])[0])

    def find_values_around(self, value: INT_FLOAT_DT64) -> list[INT_FLOAT_DT64]:
        return list(self.values[self.find_indexes(value)])

    def find_indexes_between(self,
                             start: INT_FLOAT_DT64,
                             stop: INT_FLOAT_DT64,
                             force_inclusive: bool = False
                             ) -> list[int]:

        self._raise_if_outside_range(start)
        self._raise_if_outside_range(stop)

        if start >= stop:
            raise ValueError(f"start ({start}) must be less than stop ({stop})")

        if force_inclusive:
            start = self.values[self.find_index_le(start)]
            stop = self.values[self.find_index_ge(stop)]

        return list(np.where((self.values >= start) & (self.values <= stop))[0])

    def find_values_between(self,
                            start: INT_FLOAT_DT64,
                            stop: INT_FLOAT_DT64,
                            force_inclusive: bool = False
                            ) -> list[INT_FLOAT_DT64]:

        return list(self.values[self.find_indexes_between(start,
                                                          stop,
                                                          force_inclusive)])


class AxisInt(AxisMixin, pyinterp.AxisInt64):
    pass


class AxisFloat(AxisMixin, pyinterp.Axis):
    pass


class AxisTime(AxisMixin, pyinterp.TemporalAxis):

    @staticmethod
    def _any2dt64(value: ANY2DT) -> np.datetime64:
        return np.datetime64(any2datetime(value))

    def _raise_if_outside_range(self, value: ANY2DT) -> None:
        super()._raise_if_outside_range(self._any2dt64(value))

    def find_index(self, value: ANY2DT) -> int:
        return super().find_index(self._any2dt64(value))

    def find_index_le(self, value: ANY2DT) -> int:
        return super().find_index_le(self._any2dt64(value))

    def find_index_ge(self, value: ANY2DT) -> int:
        return super().find_index_ge(self._any2dt64(value))

    def find_value_nearest(self, value: ANY2DT) -> np.datetime64:
        return super().find_value_nearest(self._any2dt64(value))

    def find_indexes(self, value: ANY2DT) -> list[int]:
        return super().find_indexes(self._any2dt64(value))

    def find_values_around(self, value: ANY2DT) -> list[np.datetime64]:
        return super().find_values_around(self._any2dt64(value))

    def find_indexes_between(self,
                             start: ANY2DT,
                             stop: ANY2DT,
                             force_inclusive: bool = False
                             ) -> list[int]:

        return super().find_indexes_between(self._any2dt64(start),
                                            self._any2dt64(stop),
                                            force_inclusive=force_inclusive)

    def find_values_between(self,
                            start: ANY2DT,
                            stop: ANY2DT,
                            force_inclusive: bool = False
                            ) -> list[np.datetime64]:

        return super().find_values_between(self._any2dt64(start),
                                           self._any2dt64(stop),
                                           force_inclusive=force_inclusive)


def array2axis(arr: np.ndarray) -> Union[AxisFloat,
                                         AxisInt,
                                         AxisTime]:

    errs = []
    for Axis in [AxisFloat, AxisInt, AxisTime]:
        try:
            axis = Axis(arr)
            break
        except Exception as e:
            errs.append(str(e))
    else:
        raise TypeError("Could not find a valid axis type. All attempts failed "
                        "with the following errors:\n" + '\n'.join(errs))

    return axis


def da2axis(da: xr.DataArray) -> Union[AxisFloat,
                                       AxisInt,
                                       AxisTime]:

    return array2axis(da.values)
