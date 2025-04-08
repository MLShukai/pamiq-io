"""Tests for the OscOutput class."""

import pytest
from pytest_mock import MockerFixture

from pamiq_io.osc.output import OscOutput


class TestOscOutput:
    """Tests for the OscOutput class."""

    @pytest.fixture
    def mock_udp_client(self, mocker: MockerFixture):
        """Creates a mock for the SimpleUDPClient."""
        mock = mocker.patch("pamiq_io.osc.output.SimpleUDPClient", autospec=True)
        return mock.return_value

    def test_init(self, mock_udp_client):
        """Tests the initialization of OscOutput."""
        # Test with default parameters
        osc_out = OscOutput()
        assert osc_out.host == "127.0.0.1"
        assert osc_out.port == 9001

        # Test with custom parameters
        custom_host = "192.168.1.100"
        custom_port = 8000
        osc_out = OscOutput(host=custom_host, port=custom_port)
        assert osc_out.host == custom_host
        assert osc_out.port == custom_port

    def test_send(self, mock_udp_client):
        """Tests sending a single OSC message."""
        osc_out = OscOutput()

        # Test with different value types
        test_cases = [
            ("/test/float", 42.0),
            ("/test/int", 42),
            ("/test/str", "hello"),
            ("/test/bool", True),
            ("/test/list", [1, 2, 3, 4]),
        ]

        for address, value in test_cases:
            osc_out.send(address, value)
            mock_udp_client.send_message.assert_called_with(address, value)

    def test_send_messages(self, mock_udp_client, mocker: MockerFixture):
        """Tests sending multiple OSC messages."""
        osc_out = OscOutput()

        # Create a test dictionary of messages
        test_messages = {
            "/slider/1": 0.5,
            "/button/toggle": True,
            "/multi/value": [1, 2, 3],
        }

        osc_out.send_messages(test_messages)

        # Assert that send_message was called for each item in the dictionary
        assert mock_udp_client.send_message.call_count == len(test_messages)

        # Check each call individually
        expected_calls = [
            mocker.call("/slider/1", 0.5),
            mocker.call("/button/toggle", True),
            mocker.call("/multi/value", [1, 2, 3]),
        ]

        mock_udp_client.send_message.assert_has_calls(expected_calls, any_order=True)
