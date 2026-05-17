
def test_pytest_setup():
    """Sanity check to ensure pytest is working."""
    assert True

def test_numpy_available():
    """Verify numpy is in the environment."""
    import numpy as np
    assert np.__version__ is not None
