from src.encapsulated_service import ServiceFacade, Request


def test_execute_success() -> None:
    facade = ServiceFacade(lambda req: {"x": req.payload["x"] * 2})
    response = facade.execute({"x": 2})

    assert response.success is True
    assert response.data == {"x": 4}
    assert response.message == "ok"


def test_execute_failure() -> None:
    def broken_handler(_: Request):
        raise ValueError("bad input")

    facade = ServiceFacade(broken_handler)
    response = facade.execute({"x": 2})

    assert response.success is False
    assert response.data == {}
    assert response.message == "bad input"
