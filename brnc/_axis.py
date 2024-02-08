#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
from functools import singledispatchmethod
import numpy as np
import xarray as xr
import pyinterp   # type: ignore

from ._types import ANY2DT, ANY2TD
from ._common import number2int, arange_inclusive, any2datetime, any2timedelta


class AxisFloat:

    def __init__(self, arr: np.ndarray) -> None:

        # using composition instead of inheritance to hide pyinterp methods
        self._ax = pyinterp.Axis(arr)

    def __repr__(self) -> str:

        step = (f"regular step {self.step}"
                if self.regular
                else "irregular steps")

        return (f"{self.__class__.__name__}: {self.values.size} elements from"
                f" {self.first} to {self.last} with {step}")

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self._ax)

    @property
    def first(self) -> float:
        return self._ax.front()

    @property
    def last(self) -> float:
        return self._ax.back()

    @property
    def firstlast(self) -> tuple[float, float]:
        return (self.first, self.last)

    @property
    def min(self) -> float:
        return self._ax.min_value()

    @property
    def max(self) -> float:
        return self._ax.max_value()

    @property
    def minmax(self) -> tuple[float, float]:
        return (self.min, self.max)

    @property
    def regular(self) -> bool:
        return self._ax.is_regular()

    @property
    def step(self) -> float:
        return self._ax.increment()

    @property
    def ascending(self) -> bool:
        return self._ax.is_ascending()

    @property
    def ascending_sign(self) -> int:
        return 1 if self.ascending else -1

    def _raise_if_outside_range(self, value: float) -> None:
        """Raise an Exception if value is outside the [min, max] range."""

        # Note: pyinterp find_index returns valid (>= 0) index for values
        # in the range [min - step/2, max + step/2]

        if self.min <= value <= self.max:
            return

        raise ValueError(f"Requested value {value} is outside the axis range "
                         f"[{self.min}, {self.max}]")

    def find_index_nearest(self, value: float) -> int:

        self._raise_if_outside_range(value)
        return self._ax.find_index(np.r_[value]).item()

    def find_index_le(self, value: float) -> int:

        idx = self.find_index_nearest(value)

        if self.values[idx] <= value:
            return idx

        return idx - 1 if self.ascending else idx + 1

    def find_index_ge(self, value: float) -> int:

        idx = self.find_index_nearest(value)

        if self.values[idx] >= value:
            return idx

        return idx + 1 if self.ascending else idx - 1

    def find_value_nearest(self, value: float) -> float:

        return self.values[self.find_index_nearest(value)]

    def find_indexes_around(self, value: float) -> list[int]:

        self._raise_if_outside_range(value)

        if len(self.values) == 1:
            raise ValueError(
                "Can't get indexes around value for an axis with length 1")

        return list(self._ax.find_indexes(np.r_[value])[0])

    def find_values_around(self, value: float) -> list[float]:

        return list(self.values[self.find_indexes_around(value)])

    def find_indexes_between(self,
                             start: float,
                             stop: float,
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
                            start: float,
                            stop: float,
                            force_inclusive: bool = False
                            ) -> list[float]:

        return list(self.values[self.find_indexes_between(start,
                                                          stop,
                                                          force_inclusive)])

    def resample_down(self, factor: int):

        # Note: np.ascontiguousarray
        # is needed to avoid error
        # TypeError: points must be a C-style contiguous array

        return self.__class__(
            np.ascontiguousarray(self.values[0::number2int(factor)]))

    def resample_up(self, factor: int):

        x = self.values

        def f(start, step, factor):
            return start + (step / factor * np.arange(factor))

        if factor == 1 or x.size == 1:
            return self

        x2 = np.vectorize(f, signature='(),(),()->(n)')(x[0:-1],
                                                        np.diff(x),
                                                        factor)

        return self.__class__(np.concatenate((x2.flatten(), [x[-1]])))

    def resample_by_step(self, step: float):

        return self.__class__(arange_inclusive(self.first,
                                               self.last,
                                               abs(step) * self.ascending_sign))

    @singledispatchmethod
    def resample(self, value):
        raise TypeError("Invalid type!")

    @resample.register(int)
    def _(self, factor: int):

        if factor >= 1:
            return self.resample_up(factor)
        elif factor <= -1:
            return self.resample_down(abs(factor))
        else:
            raise ValueError("Factor for resample_up/down must be >= 1 or <= -1")

    @resample.register(float)
    def _(self, step: float):
        return self.resample_by_step(step)

    @resample.register(np.ndarray)
    def _(self, arr: np.ndarray):
        return self.__class__(arr)


class AxisInt(AxisFloat):

    def __init__(self, arr: np.ndarray) -> None:
        # using composition instead of inheritance to hide pyinterp methods
        self._ax = pyinterp.AxisInt64(arr)

    def resample_by_step(self, step: float):
        # np.arange must be called with an int step to return a int array
        return super().resample_by_step(number2int(step))


class AxisTime(AxisFloat):

    def __init__(self, arr: np.ndarray) -> None:
        # using composition instead of inheritance to hide pyinterp methods
        self._ax = pyinterp.TemporalAxis(arr)

    @property
    def step(self) -> np.timedelta64:

        # get time step
        # Note: xarray uses datetime64[ns] and timedelta64[ns]
        time_delta = np.timedelta64(super().step)

        # allow only time frequencies with constant time duration, not
        # M (month, 28 to 31 days) or Y (year, 365 or 366 days), as this can
        # result in:
        # UFuncTypeError: Cannot cast ufunc 'add' input 1 from dtype('<m8[M]')
        # to dtype('<m8[D]') with casting rule 'same_kind'
        units = ["W", "D", "h", "m", "s", "ms", "us", "ns"]

        if np.datetime_data(time_delta)[0] not in units:
            raise ValueError(
                f"Invalid unit for time step '{time_delta}', must be one of: "
                + ", ".join(units))

        # find the largest possible time unit without precision loss
        for unit in units:
            time_delta2 = np.timedelta64(time_delta, unit)
            if time_delta2 == time_delta:
                return time_delta2
        else:
            # this should never happen
            raise ValueError("Couldn't convert np.timedelta64 to a 'nice' unit")

    @staticmethod
    def _any2dt64(value: ANY2DT) -> np.datetime64:
        return np.datetime64(any2datetime(value))

    @staticmethod
    def _any2td64(value: ANY2TD) -> np.timedelta64:
        return np.timedelta64(any2timedelta(value))

    def _raise_if_outside_range(self, value: ANY2DT) -> None:
        super()._raise_if_outside_range(self._any2dt64(value))

    def find_index_nearest(self, value: ANY2DT) -> int:
        return super().find_index_nearest(self._any2dt64(value))

    def find_index_le(self, value: ANY2DT) -> int:
        return super().find_index_le(self._any2dt64(value))

    def find_index_ge(self, value: ANY2DT) -> int:
        return super().find_index_ge(self._any2dt64(value))

    def find_value_nearest(self, value: ANY2DT) -> np.datetime64:
        return super().find_value_nearest(self._any2dt64(value))

    def find_indexes_around(self, value: ANY2DT) -> list[int]:
        return super().find_indexes_around(self._any2dt64(value))

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

    def resample_by_step(self, step: ANY2TD):

        return self.__class__(arange_inclusive(
            self.first,
            self.last,
            abs(self._any2td64(step)) * self.ascending_sign))


class AxisFactory:

    def from_array(self, arr: np.ndarray) -> Union[AxisFloat,
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

    def from_dataarray(self, da: xr.DataArray) -> Union[AxisFloat,
                                                        AxisInt,
                                                        AxisTime]:

        return self.from_array(da.values)
