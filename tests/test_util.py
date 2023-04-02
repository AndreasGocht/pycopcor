import pytest
import pycopcor.utils


def test_no_bindings():
    test_array = [1, 2, 3, 4, 5]
    bindings_found, min_distance, bindings = pycopcor.utils.bindings_and_min_distance(test_array)
    assert bindings_found == False
    assert min_distance == 1.0
    assert all(bindings == [0, 0, 0, 0, 0])


def test_bindings():
    test_array = [1, 2, 1, 4, 5]
    bindings_found, min_distance, bindings = pycopcor.utils.bindings_and_min_distance(test_array)
    assert bindings_found
    assert min_distance == 1.0
    assert all(bindings == [1, 0, 1, 0, 0])
