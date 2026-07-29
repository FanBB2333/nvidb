#!/bin/sh
# nvidb queue keeper - keeps `nvidb queue daemon` running on this machine.
#
# Written here by `nvidb queue keeper install`. It is a plain file in the nvidb
# working directory on purpose: read it, edit it, or run it by hand. Nothing
# else on the machine has to know it exists - no cron entry, no service unit -
# so it needs neither root nor a configured login session.
#
# A reboot is deliberately not survived. Whoever next talks to the queue runs
# `keeper ensure`, which starts it again and does nothing when it is already up.
set -u

NVIDB_HOME="${NVIDB_HOME:-__NVIDB_HOME__}"
# Baked in as an absolute path at install time: `ssh host 'nvidb ...'` gets a
# non-interactive shell, and on most distributions that never reads the profile
# lines which put a `pip install --user` binary on PATH.
NVIDB="${NVIDB_BIN:-__NVIDB_BIN__}"
INTERVAL="${NVIDB_QUEUE_INTERVAL:-__INTERVAL__}"

PIDFILE="$NVIDB_HOME/queue-keeper.pid"
LOCKDIR="$NVIDB_HOME/queue-keeper.lock"
SESSION="$NVIDB_HOME/queue-keeper.session"
LOG="$NVIDB_HOME/queue-keeper.log"

alive() {
  [ -f "$PIDFILE" ] || return 1
  kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null
}

case "${1:-start}" in
  start|ensure)
    alive && exit 0
    mkdir -p "$NVIDB_HOME" || exit 1
    # A lock directory surviving without a live keeper is debris from a kill -9
    # or a power cut, never a keeper that is still starting.
    rmdir "$LOCKDIR" 2>/dev/null
    # mkdir is atomic on POSIX, so when several clients race to start the keeper
    # exactly one proceeds. The losers exit quietly - the winner is starting it.
    mkdir "$LOCKDIR" 2>/dev/null || exit 0
    if command -v setsid >/dev/null 2>&1; then
      # Its own session, so the keeper outlives the SSH channel that started it
      # and can later be signalled as a process group.
      echo 1 > "$SESSION"
      setsid "$0" _loop >> "$LOG" 2>&1 < /dev/null &
    else
      # macOS has no setsid. nohup still detaches, but the keeper then shares a
      # process group with the calling shell and must never be signalled by
      # group id - which would take the caller down with it.
      echo 0 > "$SESSION"
      nohup "$0" _loop >> "$LOG" 2>&1 < /dev/null &
    fi
    exit 0
    ;;
  _loop)
    echo $$ > "$PIDFILE"
    rmdir "$LOCKDIR" 2>/dev/null
    daemon_pid=""
    # The trap must end in `exit`: a handler that only cleans up returns into
    # the loop below, which would start a fresh daemon seconds after being
    # asked to stop. Taking the daemon with it explicitly matters too, since
    # without setsid there is no process group that means "just these two".
    nvidb_shutdown() {
      rm -f "$PIDFILE"
      [ -n "$daemon_pid" ] && kill -TERM "$daemon_pid" 2>/dev/null
      echo "[keeper] $(date) stopped"
      exit 143
    }
    trap nvidb_shutdown TERM INT
    echo "[keeper] $(date) up, interval=${INTERVAL}s, nvidb=$NVIDB"
    while : ; do
      # Backgrounded and waited on, rather than run in the foreground: a signal
      # arriving during `wait` runs the trap at once, where one arriving while a
      # foreground child runs would be held until that child exited on its own.
      "$NVIDB" queue daemon --interval "$INTERVAL" &
      daemon_pid=$!
      wait "$daemon_pid"
      rc=$?
      daemon_pid=""
      # Restarting at full speed would spin when the daemon cannot start at all
      # - a broken install, an unreadable config - and bury the reason in log.
      # The pause makes that failure cheap to survive and easy to read.
      echo "[keeper] $(date) daemon exited ($rc), restarting in 5s"
      sleep 5
    done
    ;;
  stop)
    if alive; then
      pid=$(cat "$PIDFILE")
      if [ "$(cat "$SESSION" 2>/dev/null)" = "1" ]; then
        # The whole group: the keeper and the daemon it supervises.
        kill -TERM "-$pid" 2>/dev/null
      else
        kill -TERM "$pid" 2>/dev/null
        pkill -TERM -P "$pid" 2>/dev/null
      fi
    fi
    rm -f "$PIDFILE"
    rmdir "$LOCKDIR" 2>/dev/null
    echo "stopped"
    ;;
  restart)
    "$0" stop > /dev/null 2>&1
    sleep 1
    exec "$0" start
    ;;
  status)
    if alive; then
      echo "running (pid $(cat "$PIDFILE"))"
    else
      echo "stopped"
      exit 1
    fi
    ;;
  logs)
    tail -n "${2:-50}" "$LOG" 2>/dev/null
    ;;
  *)
    echo "usage: $0 {start|ensure|stop|restart|status|logs [LINES]}" >&2
    exit 2
    ;;
esac
