"""video-link 状态引擎跨切片共享异常。"""

from __future__ import annotations

from http import HTTPStatus


class BridgeError(Exception):
    """可映射为 HTTP 状态的引擎层错误。"""

    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message