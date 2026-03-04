"""通用业务逻辑封装示例。

由于当前仓库没有现有业务代码，这里先提供一个可复用的封装骨架：
- 对外暴露统一入口 `ServiceFacade.execute`
- 使用 `Request` / `Response` 数据模型约束输入输出
- 通过可注入处理器实现业务逻辑与调用方解耦
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class Request:
    """统一请求对象。"""

    payload: Dict[str, Any]


@dataclass(frozen=True)
class Response:
    """统一响应对象。"""

    success: bool
    data: Dict[str, Any]
    message: str = ""


class ServiceFacade:
    """统一封装入口。

    你可以把已有的零散函数逻辑放入 `handler`，
    外部只通过 `execute` 调用，降低耦合。
    """

    def __init__(self, handler: Callable[[Request], Dict[str, Any]]) -> None:
        self._handler = handler

    def execute(self, payload: Dict[str, Any]) -> Response:
        request = Request(payload=payload)
        try:
            result = self._handler(request)
            return Response(success=True, data=result, message="ok")
        except Exception as exc:  # 业务异常统一收口
            return Response(success=False, data={}, message=str(exc))


def default_handler(request: Request) -> Dict[str, Any]:
    """默认处理器示例：回显输入。"""

    return {"echo": request.payload}
