#!/usr/bin/env python3
"""Keep the cluster's default model loaded.

~/exo-autoload.txt holds one line, "<model_id> <node_count>", for example:

    mlx-community/MiMo-V2.6-Pro-RL-mxfp4-q8 3

Whenever exo has had no instance at all for a minute, this launches that model
on that many nodes (the RDMA pipeline placement when node_count is above 1).
It does nothing while the file is missing or empty, so empty the file before
loading a different model by hand.

exo-run.sh starts this next to exo on every node. Only a node with the file
acts. Standard library only, so any python3 runs it.
"""

import json
import os
import time
import urllib.request
from pathlib import Path

API = "http://localhost:52415"
CONF = Path.home() / "exo-autoload.txt"
MAX_LAUNCHES_PER_HOUR = 3  # a model that keeps dying is not relaunched forever


def get(path):
    with urllib.request.urlopen(API + path, timeout=30) as response:
        return json.load(response)


def launch(model, nodes):
    """Create the instance if exo offers that placement yet. True if launched."""
    for preview in get("/instance/previews?model_id=" + model)["previews"]:
        if (
            preview["sharding"] == "Pipeline"
            and (nodes == 1 or preview["instance_meta"] == "MlxJaccl")
            and not preview["error"]
            and len(preview["memory_delta_by_node"] or {}) == nodes
        ):
            request = urllib.request.Request(
                API + "/instance",
                json.dumps({"instance": preview["instance"]}).encode(),
                {"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=30).read()
            return True
    return False


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def main():
    exo_pid = os.getppid()  # exo-run.sh, which becomes exo
    idle_checks = 0
    launches = []  # times of this hour's launches
    while True:
        time.sleep(30)
        if os.getppid() != exo_pid:
            return  # that exo is gone; its replacement starts its own copy of this
        try:
            wanted = CONF.read_text().split() if CONF.exists() else []
            if len(wanted) != 2 or get("/state/instances"):
                idle_checks = 0
                continue
            idle_checks += 1
            launches = [t for t in launches if time.time() - t < 3600]
            if idle_checks < 2 or len(launches) >= MAX_LAUNCHES_PER_HOUR:
                continue
            model, nodes = wanted[0], int(wanted[1])
            if launch(model, nodes):
                launches.append(time.time())
                idle_checks = 0
                log(f"launched {model} on {nodes} node(s), {len(launches)} this hour")
            else:
                log(f"waiting for a {nodes} node placement of {model}")
        except (OSError, ValueError, KeyError) as error:
            log(f"waiting: {error}")  # exo still starting, or a bad config line


if __name__ == "__main__":
    main()
