"""A rebuilt dashboard is served, not a cached page from the previous build."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.dashboard_files import DashboardFiles


def dashboard(directory: Path, page: str) -> TestClient:
    (directory / "_app").mkdir(exist_ok=True)
    (directory / "index.html").write_text(page)
    (directory / "_app" / "start.js").write_text("console.log(1)")
    # Like the Nix store: every file has the same modification time
    for path in (directory / "index.html", directory / "_app" / "start.js"):
        os.utime(path, (1, 1))
    app = FastAPI()
    app.mount("/", DashboardFiles(directory=directory, html=True), name="dashboard")
    return TestClient(app)


def test_a_rebuilt_page_of_the_same_size_is_sent(tmp_path: Path) -> None:
    old = dashboard(tmp_path, '<script src="/_app/a1b2.js"></script>').get("/")
    new_client = dashboard(tmp_path, '<script src="/_app/c3d4.js"></script>')

    # The browser asks whether its copy of the old build's page is still current
    response = new_client.get("/", headers={"if-none-match": old.headers["etag"]})

    assert response.status_code == 200
    assert "c3d4" in response.text
    assert response.headers["cache-control"] == "no-store"


def test_scripts_can_still_be_cached(tmp_path: Path) -> None:
    client = dashboard(tmp_path, "<html></html>")
    first = client.get("/_app/start.js")

    response = client.get(
        "/_app/start.js", headers={"if-none-match": first.headers["etag"]}
    )

    assert response.status_code == 304
    assert "cache-control" not in first.headers
