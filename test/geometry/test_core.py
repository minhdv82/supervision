import pytest

from supervision.geometry.core import Point, Vec4, Vector


@pytest.mark.parametrize(
    "vector, point, expected_result",
    [
        (Vector(start=Point(x=0, y=0), end=Point(x=5, y=5)), Point(x=-1, y=1), 10.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=5, y=5)), Point(x=6, y=6), 0.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=5, y=5)), Point(x=3, y=6), 15.0),
        (Vector(start=Point(x=5, y=5), end=Point(x=0, y=0)), Point(x=-1, y=1), -10.0),
        (Vector(start=Point(x=5, y=5), end=Point(x=0, y=0)), Point(x=6, y=6), 0.0),
        (Vector(start=Point(x=5, y=5), end=Point(x=0, y=0)), Point(x=3, y=6), -15.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=1, y=0)), Point(x=0, y=0), 0.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=1, y=0)), Point(x=0, y=-1), -1.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=1, y=0)), Point(x=0, y=1), 1.0),
        (Vector(start=Point(x=1, y=0), end=Point(x=0, y=0)), Point(x=0, y=0), 0.0),
        (Vector(start=Point(x=1, y=0), end=Point(x=0, y=0)), Point(x=0, y=-1), 1.0),
        (Vector(start=Point(x=1, y=0), end=Point(x=0, y=0)), Point(x=0, y=1), -1.0),
        (Vector(start=Point(x=1, y=1), end=Point(x=1, y=3)), Point(x=0, y=0), 2.0),
        (Vector(start=Point(x=1, y=1), end=Point(x=1, y=3)), Point(x=1, y=4), 0.0),
        (Vector(start=Point(x=1, y=1), end=Point(x=1, y=3)), Point(x=2, y=4), -2.0),
        (Vector(start=Point(x=1, y=3), end=Point(x=1, y=1)), Point(x=0, y=0), -2.0),
        (Vector(start=Point(x=1, y=3), end=Point(x=1, y=1)), Point(x=1, y=4), 0.0),
        (Vector(start=Point(x=1, y=3), end=Point(x=1, y=1)), Point(x=2, y=4), 2.0),
    ],
)
def test_vector_cross_product(
    vector: Vector, point: Point, expected_result: float
) -> None:
    result = vector.cross_product(point=point)
    assert result == expected_result


@pytest.mark.parametrize(
    "vector, expected_result",
    [
        (Vector(start=Point(x=0, y=0), end=Point(x=0, y=0)), 0.0),
        (Vector(start=Point(x=1, y=0), end=Point(x=0, y=0)), 1.0),
        (Vector(start=Point(x=0, y=1), end=Point(x=0, y=0)), 1.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=1, y=0)), 1.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=0, y=1)), 1.0),
        (Vector(start=Point(x=-1, y=0), end=Point(x=0, y=0)), 1.0),
        (Vector(start=Point(x=0, y=-1), end=Point(x=0, y=0)), 1.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=-1, y=0)), 1.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=0, y=-1)), 1.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=3, y=4)), 5.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=-3, y=4)), 5.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=3, y=-4)), 5.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=-3, y=-4)), 5.0),
        (Vector(start=Point(x=0, y=0), end=Point(x=4, y=3)), 5.0),
        (Vector(start=Point(x=3, y=4), end=Point(x=0, y=0)), 5.0),
        (Vector(start=Point(x=4, y=3), end=Point(x=0, y=0)), 5.0),
    ],
)
def test_vector_magnitude(vector: Vector, expected_result: float) -> None:
    result = vector.magnitude
    assert result == expected_result


@pytest.mark.parametrize(
    "vec, expected_result",
    [
        (Vec4(t=0, x=1, y=1, z=0), -2),
        (Vec4(t=1, x=1, y=1, z=0), -1),
        (Vec4(t=1, x=0, y=1, z=0), -0),
        (Vec4(t=2, x=0, y=1, z=0), 3),
    ],
)
def test_four_vector_magnitude(vec: Vec4, expected_result: float) -> None:
    assert vec.minkowski == expected_result


@pytest.mark.parametrize(
    "vec, expected_result",
    [
        (Vec4(t=0, x=1, y=1, z=0), False),
        (Vec4(t=1, x=1, y=1, z=0), False),
        (Vec4(t=1, x=0, y=1, z=0), False),
        (Vec4(t=2, x=0, y=1, z=0), True),
    ],
)
def test_four_vector_timelike(vec: Vec4, expected_result: bool) -> None:
    assert vec.timelike == expected_result


@pytest.mark.parametrize(
    "vec, expected_result",
    [
        (Vec4(t=0, x=1, y=1, z=0), True),
        (Vec4(t=1, x=1, y=1, z=0), True),
        (Vec4(t=1, x=0, y=1, z=0), False),
        (Vec4(t=2, x=0, y=1, z=0), False),
    ],
)
def test_four_vector_spacelike(vec: Vec4, expected_result: bool) -> None:
    assert vec.spacelike == expected_result


@pytest.mark.parametrize(
    "vec, expected_result",
    [
        (Vec4(t=0, x=1, y=1, z=0), False),
        (Vec4(t=2, x=2, y=0, z=0), True),
        (Vec4(t=1, x=0, y=1, z=0), True),
        (Vec4(t=2, x=0, y=1, z=0), False),
    ],
)
def test_four_vector_lightlike(vec: Vec4, expected_result: bool) -> None:
    assert vec.lightlike == expected_result


@pytest.mark.parametrize(
    "lhs, rhs, expected_result",
    [
        (Vec4(t=0, x=1, y=1, z=0), Vec4(t=0, x=1, y=1, z=0), -2),
        (Vec4(t=1, x=1, y=1, z=0), Vec4(t=0, x=1, y=1, z=0), -2),
        (Vec4(t=2, x=1, y=1, z=0), Vec4(t=1, x=1, y=1, z=0), 0),
        (Vec4(t=2, x=1, y=1, z=1), Vec4(t=2, x=1, y=1, z=0), 2),
    ],
)
def test_four_vector_multiplication(
    lhs: Vec4, rhs: Vec4, expected_result: float
) -> None:
    return lhs * rhs == expected_result
