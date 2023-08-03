"""
This submodule contains functions that contain business unit-specific logic.

As it is not necessary to load the entire 'segment' submodule, users must specify which business unit (submodule of 'segment') they wish to load. For example:
```python
import dscf.segment.mvh
```

Functions that are specific to the Mars Petcare Data Platform (PDP), e.g. relating to the PDP's data products, should be placed within the relevant part of dscf's submodule 'petcare'.
"""

# The following has been commented out as it is not necessary to load the entire 'segment' submodule.
# Instead, users must specify which business unit (submodule of 'segment') they wish to load.

# ----- Load all functions in seperate .py files within each sub-directory as one class. ----- #
__all__ = []

import pkgutil
import inspect

for loader, name, is_pkg in pkgutil.walk_packages(__path__):
    module = loader.find_module(name).load_module(name)

    for name, value in inspect.getmembers(module):
        if name.startswith('__'):
            continue

        globals()[name] = value
        __all__.append(name)