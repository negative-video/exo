#!/bin/zsh
# Runs inside the SSH session opened by exo-service.sh, so exo gets the same
# environment as when it is started by hand (~/.zshrc: HF_TOKEN, PATH).

# Keeps the default model loaded on the node that has ~/exo-autoload.txt.
# It exits by itself once the exo below is gone.
python3 ~/exo/service/exo-autoload.py >> ~/Library/Logs/exo-autoload.log 2>&1 &

exec ~/exo-result/bin/exo >> ~/Library/Logs/exo.log 2>&1
