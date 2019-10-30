import pytest
import doctest
import neurokit2 as nk


if __name__ == '__main__':
    doctest.testmod()
    pytest.main()



def test_foo():
    assert nk.foo() == 4
