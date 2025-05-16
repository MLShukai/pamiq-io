import pytest

from pamiq_io.keyboard.output.base import Key, KeyboardOutput


class TestKeyboardOutput:
    @pytest.mark.parametrize("method", ["press", "release"])
    def test_abstract_method(self, method):
        assert method in KeyboardOutput.__abstractmethods__


import pytest

from pamiq_io.keyboard.output.base import Key


class TestKeyProperties:
    """Tests for Key enum property groups."""

    def test_number_keys(self):
        """Test that number_keys returns all number keys and only number
        keys."""
        number_keys = Key.number_keys()

        # Check return type
        assert isinstance(number_keys, set)

        # Check contents
        assert Key.KEY_0 in number_keys
        assert Key.KEY_1 in number_keys
        assert Key.KEY_9 in number_keys
        assert len(number_keys) == 10

        # Check non-number keys are excluded
        assert Key.A not in number_keys
        assert Key.F1 not in number_keys

    def test_function_keys(self):
        """Test that function_keys returns all function keys and only function
        keys."""
        function_keys = Key.function_keys()

        # Check return type
        assert isinstance(function_keys, set)

        # Check contents
        assert Key.F1 in function_keys
        assert Key.F5 in function_keys
        assert Key.F12 in function_keys
        assert len(function_keys) == 12

        # Check non-function keys are excluded
        assert Key.A not in function_keys
        assert Key.KEY_1 not in function_keys

    def test_letter_keys(self):
        """Test that letter_keys returns all letter keys and only letter
        keys."""
        letter_keys = Key.letter_keys()

        # Check return type
        assert isinstance(letter_keys, set)

        # Check contents
        assert Key.A in letter_keys
        assert Key.Z in letter_keys
        assert len(letter_keys) == 26

        # Check non-letter keys are excluded
        assert Key.F1 not in letter_keys
        assert Key.KEY_1 not in letter_keys
        assert Key.SPACE not in letter_keys

    def test_super_keys(self):
        """Test that super_keys returns all super keys and only super keys."""
        super_keys = Key.super_keys()

        # Check return type
        assert isinstance(super_keys, set)

        # Check contents
        assert Key.LEFT_SUPER in super_keys
        assert Key.RIGHT_SUPER in super_keys
        assert len(super_keys) == 2

        # Check non-super keys are excluded
        assert Key.CTRL not in super_keys
        assert Key.ALT not in super_keys

    def test_shift_keys(self):
        """Test that shift_keys returns all shift keys and only shift keys."""
        shift_keys = Key.shift_keys()

        # Check return type
        assert isinstance(shift_keys, set)

        # Check contents
        assert Key.SHIFT in shift_keys
        assert Key.LEFT_SHIFT in shift_keys
        assert Key.RIGHT_SHIFT in shift_keys
        assert len(shift_keys) == 3

        # Check non-shift keys are excluded
        assert Key.CTRL not in shift_keys
        assert Key.ALT not in shift_keys

    def test_alt_keys(self):
        """Test that alt_keys returns all alt keys and only alt keys."""
        alt_keys = Key.alt_keys()

        # Check return type
        assert isinstance(alt_keys, set)

        # Check contents
        assert Key.ALT in alt_keys
        assert Key.LEFT_ALT in alt_keys
        assert Key.RIGHT_ALT in alt_keys
        assert len(alt_keys) == 3

        # Check non-alt keys are excluded
        assert Key.CTRL not in alt_keys
        assert Key.SHIFT not in alt_keys

    def test_ctrl_keys(self):
        """Test that ctrl_keys returns all control keys and only control
        keys."""
        ctrl_keys = Key.ctrl_keys()

        # Check return type
        assert isinstance(ctrl_keys, set)

        # Check contents
        assert Key.CTRL in ctrl_keys
        assert Key.LEFT_CONTROL in ctrl_keys
        assert Key.RIGHT_CONTROL in ctrl_keys
        assert len(ctrl_keys) == 3

        # Check non-control keys are excluded
        assert Key.ALT not in ctrl_keys
        assert Key.SHIFT not in ctrl_keys
