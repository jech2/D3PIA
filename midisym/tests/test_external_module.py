from midisym import mymodule, mymodule2


def test_external_module():
    assert mymodule.add(1, 2) == 3


def test_external_module2():
    print(mymodule2.say_hello())
