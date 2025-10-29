from typing import Any, Dict, Callable, NewType
from app.helpers.miscellaneous import ParametersDict

LayoutDictTypes = NewType('LayoutDictTypes', Dict[str, Dict[str, Dict[str, Any]]])

ParametersTypes = NewType('ParametersTypes', ParametersDict)
FacesParametersTypes = NewType('FacesParametersTypes', dict[int, ParametersTypes])

ControlTypes = NewType('ControlTypes', Dict[str, bool|int|float|str|dict])

MarkerTypes = NewType('MarkerTypes', Dict[int, Dict[str, FacesParametersTypes|ControlTypes]])