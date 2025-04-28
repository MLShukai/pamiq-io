"""Tests for the OscInput class."""

import threading
import time

import pytest
from pytest_mock import MockerFixture

from pamiq_io.osc.input import OscInput


class TestOscInput:
    """Tests for the OscInput class."""

    @pytest.fixture
    def mock_dispatcher(self, mocker: MockerFixture):
        """Creates a mock for the Dispatcher."""
        return mocker.patch("pamiq_io.osc.input.Dispatcher", autospec=True).return_value

    @pytest.fixture
    def mock_server(self, mocker: MockerFixture):
        """Creates a mock for the ThreadingOSCUDPServer."""
        server_mock = mocker.patch(
            "pamiq_io.osc.input.ThreadingOSCUDPServer", autospec=True
        )
        return server_mock.return_value

    def test_init(self):
        """Tests the initialization of OscInput."""
        # Test with default parameters
        osc_in = OscInput()
        assert osc_in.host == "127.0.0.1"
        assert osc_in.port == 9001

        # Test with custom parameters
        custom_host = "192.168.1.100"
        custom_port = 8000
        osc_in = OscInput(host=custom_host, port=custom_port)
        assert osc_in.host == custom_host
        assert osc_in.port == custom_port

    def test_add_handler(self, mock_dispatcher):
        """Tests adding a handler function for an OSC address pattern."""

        def test_handler(addr: str, *args):
            pass

        osc_in = OscInput()
        osc_in.add_handler("/test", test_handler)

        # Verify that map was called on the dispatcher
        mock_dispatcher.map.assert_called_once()
        # Check that the first argument to map is the correct address pattern
        assert mock_dispatcher.map.call_args[0][0] == "/test"

    def test_start_blocking(self, mock_server):
        """Tests starting the OSC server in blocking mode."""
        osc_in = OscInput()

        # Set up mock server to raise an exception when serve_forever is called
        # This is to exit the blocking call for testing
        mock_server.serve_forever.side_effect = KeyboardInterrupt

        # Test blocking mode
        try:
            osc_in.start(blocking=True)
        except KeyboardInterrupt:
            pass

        # Verify serve_forever was called
        mock_server.serve_forever.assert_called_once()

    def test_start_nonblocking(self, mock_server):
        """Tests starting the OSC server in non-blocking mode."""
        osc_in = OscInput()
        osc_in.start(blocking=False)

        # Verify that a server thread was created and started
        osc_in.stop()

    def test_stop(self, mock_server):
        """Tests stopping the OSC server."""
        osc_in = OscInput()

        # Start in non-blocking mode
        osc_in.start(blocking=False)

        # Wait briefly to ensure thread starts
        time.sleep(0.1)

        # Now stop the server
        osc_in.stop()

        # Verify shutdown was called
        mock_server.shutdown.assert_called_once()

    def test_start_already_running_error(self, mock_server):
        """Tests error when trying to start an already running server."""
        osc_in = OscInput()

        # Start the server
        osc_in.start(blocking=False)

        # Try starting again and expect RuntimeError
        with pytest.raises(RuntimeError, match="OSC input server is already running"):
            osc_in.start()

        # Clean up
        osc_in.stop()

    def test_stop_not_running_error(self):
        """Tests error when trying to stop a server that is not running."""
        osc_in = OscInput()

        # Try stopping without starting and expect RuntimeError
        with pytest.raises(RuntimeError, match="OSC input server is not running"):
            osc_in.stop()

    def test_cleanup_on_deletion(self, mock_server):
        """Tests that server is properly closed on object deletion."""
        osc_in = OscInput()

        # Start the server
        osc_in.start(blocking=False)

        # Mock the stop method to check if it's called during deletion
        original_stop = osc_in.stop
        stop_called = False

        def mock_stop():
            nonlocal stop_called
            stop_called = True
            original_stop()

        osc_in.stop = mock_stop

        # Trigger deletion
        osc_in.__del__()

        # Verify stop was called
        assert stop_called
