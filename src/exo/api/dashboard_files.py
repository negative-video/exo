import os
from typing import final, override

from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.responses import Response
from starlette.staticfiles import PathLike
from starlette.types import Scope


def _is_page(headers: Headers) -> bool:
    return headers.get("content-type", "").startswith("text/html")


@final
class DashboardFiles(StaticFiles):
    """The dashboard's files, with its pages never cached.

    A file's ETag comes from its modification time and size. Nix gives every file in the
    store the same modification time, so a rebuilt page of the same size has the ETag of
    the page it replaces: the browser keeps the old page, which asks for scripts the new
    build doesn't have, and shows nothing. Scripts and styles have content-hashed names,
    so they can still be cached.
    """

    @override
    def file_response(
        self,
        full_path: PathLike,
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        if _is_page(response.headers):
            response.headers["cache-control"] = "no-store"
        return response

    @override
    def is_not_modified(
        self, response_headers: Headers, request_headers: Headers
    ) -> bool:
        # A page's ETag can't tell two builds apart, so a page is always sent
        if _is_page(response_headers):
            return False
        return super().is_not_modified(response_headers, request_headers)
