"""Tests for the DirectInputMouseOutput class."""

from tests.helpers import skip_if_platform_is_not_windows

skip_if_platform_is_not_windows()

import pytest
from pytest_mock import MockerFixture

from pamiq_io.mouse.output.directinput import DirectInputMouseOutput, MouseButton


class TestDirectInputMouseOutput:
    """Tests for the DirectInputMouseOutput class."""

    @pytest.fixture
    def mock_directinput(self, mocker: MockerFixture):
        """Create a mock for the pydirectinput module."""
        mock = mocker.patch("pamiq_io.mouse.output.directinput.pydirectinput")
        # Set up PRIMARY and SECONDARY attributes
        mock.PRIMARY = "x1"
        mock.SECONDARY = "x2"
        return mock

    def test_init_sets_pause(self, mock_directinput):
        """Test that init properly sets pydirectinput.PAUSE."""
        fps = 30.0
        DirectInputMouseOutput(fps=fps)
        assert mock_directinput.PAUSE == 1 / fps

    def test_convert_to_directinput_button(self, mock_directinput):
        """Test converting to DirectInput button strings."""
        # Test regular buttons
        assert (
            DirectInputMouseOutput.convert_to_directinput_button(MouseButton.LEFT)
            == "left"
        )
        assert (
            DirectInputMouseOutput.convert_to_directinput_button(MouseButton.RIGHT)
            == "right"
        )
        assert (
            DirectInputMouseOutput.convert_to_directinput_button(MouseButton.MIDDLE)
            == "middle"
        )

        # Test special case buttons
        assert (
            DirectInputMouseOutput.convert_to_directinput_button(MouseButton.SIDE)
            == mock_directinput.PRIMARY
        )
        assert (
            DirectInputMouseOutput.convert_to_directinput_button(MouseButton.EXTRA)
            == mock_directinput.SECONDARY
        )

    def test_press(self, mock_directinput):
        """Test pressing a mouse button."""
        mouse_output = DirectInputMouseOutput()

        # Test with standard button
        mouse_output.press(MouseButton.RIGHT)
        mock_directinput.mouseDown.assert_called_with(button="right")

        # Test with special button
        mouse_output.press(MouseButton.SIDE)
        mock_directinput.mouseDown.assert_called_with(button=mock_directinput.PRIMARY)

    def test_release(self, mock_directinput):
        """Test releasing a mouse button."""
        mouse_output = DirectInputMouseOutput()

        # Test with standard button
        mouse_output.release(MouseButton.MIDDLE)
        mock_directinput.mouseUp.assert_called_with(button="middle")

        # Test with special button
        mouse_output.release(MouseButton.EXTRA)
        mock_directinput.mouseUp.assert_called_with(button=mock_directinput.SECONDARY)
