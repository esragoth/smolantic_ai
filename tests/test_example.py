"""
Tests for the example module.
"""

from smolantic_ai.example import hello_world

def test_hello_world():
    """Test that hello_world returns the expected message."""
    assert hello_world() == "Hello from Smolantic AI!" 