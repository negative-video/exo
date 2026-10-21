#!/bin/zsh
# Runs inside the SSH session opened by exo-service.sh, so exo gets the same
# environment as when it is started by hand (~/.zshrc: HF_TOKEN, PATH).

# These logs are only ever appended to, so keep just the previous run's and start
# afresh: a record of when the cluster was used shouldn't build up forever. (exo
# rotates its own ~/.exo/exo_log/exo.log.)
for log in ~/Library/Logs/exo.log ~/Library/Logs/exo-autoload.log \
  ~/.exo/exo_log/runner_log/stdout.log ~/.exo/exo_log/runner_log/stderr.log; do
  [ -f $log ] && mv -f $log $log.1
done

# Keeps the default model loaded on the node that has ~/exo-autoload.txt.
# It exits by itself once the exo below is gone.
python3 ~/exo/service/exo-autoload.py >> ~/Library/Logs/exo-autoload.log 2>&1 &

exec ~/exo-result/bin/exo >> ~/Library/Logs/exo.log 2>&1
