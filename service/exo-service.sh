#!/bin/zsh
# launchd entry point for exo (com.negative-video.exo). Same on every node.
#
# exo is started through an SSH session to localhost on purpose. macOS local
# network privacy blocks multicast from a launchd job that runs as a user, and
# exo finds its peers by multicast, so an exo started directly by launchd only
# ever sees itself. Tools started over SSH are exempt (Apple TN3179), which is
# why exo run by hand from an SSH shell has always formed the cluster.

# /nix is a separate volume that mounts late at boot. Wait for the build.
until [ -x ~/exo-result/bin/exo ]; do sleep 2; done

# -tt gives exo a terminal, so stopping this job hangs it up and exo shuts
# down cleanly. If ssh or exo exits, launchd starts this script again.
TERM=dumb exec ssh -tt -i ~/.ssh/exo_local -o BatchMode=yes localhost 'zsh -lic ~/exo/service/exo-run.sh'
