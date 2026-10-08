#!/bin/zsh
# One-time setup per node. Makes exo start at boot and restart if it dies.
# Stop any hand-started exo first. Safe to run again.
set -e

# A key this node uses only to SSH into itself (see exo-service.sh for why).
key=~/.ssh/exo_local
[ -f $key ] || ssh-keygen -q -t ed25519 -N '' -C exo-service -f $key
grep -qF "$(cat $key.pub)" ~/.ssh/authorized_keys 2>/dev/null ||
    echo "from=\"127.0.0.1,::1\" $(cat $key.pub)" >> ~/.ssh/authorized_keys
ssh -i $key -o BatchMode=yes -o StrictHostKeyChecking=accept-new localhost true

# A leftover login item from an early recovery session starts a second exo at
# login, which would take the ports before the service gets them.
stray=~/Library/LaunchAgents/net.exolabs.exo.plist
if [ -f $stray ]; then
    launchctl bootout gui/$(id -u)/net.exolabs.exo 2>/dev/null || true
    mv $stray ~/net.exolabs.exo.plist.disabled
fi

plist=/Library/LaunchDaemons/com.negative-video.exo.plist
sudo launchctl bootout system/com.negative-video.exo 2>/dev/null || true
sudo tee $plist > /dev/null <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.negative-video.exo</string>
    <key>UserName</key><string>$USER</string>
    <key>ProgramArguments</key>
    <array><string>$HOME/exo/service/exo-service.sh</string></array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>$HOME/Library/Logs/exo-service.log</string>
    <key>StandardErrorPath</key><string>$HOME/Library/Logs/exo-service.log</string>
</dict>
</plist>
EOF
sudo launchctl bootstrap system $plist
echo "exo service installed on $(hostname -s)"
