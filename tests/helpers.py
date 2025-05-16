import sys

import pytest


def skip_if_platform_is_not_linux():
    if sys.platform != "linux":
        pytest.skip("Platform is not linux.", allow_module_level=True)


def skip_if_platform_is_not_windows():
    if sys.platform != "win32":
        pytest.skip("Platform is not win32.", allow_module_level=True)
