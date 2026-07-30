#!/bin/sh
# nvidb queue keeper - keeps `nvidb queue daemon` running on this machine.
#
# Shell mode is self-contained and needs no root or service manager. Systemd
# mode delegates lifecycle operations to an optional user unit, which can start
# at login (or at boot when lingering is enabled).
set -u

DEFAULT_NVIDB_HOME=__NVIDB_HOME__
DEFAULT_NVIDB=__NVIDB_BIN__
NVIDB_HOME="${NVIDB_HOME:-$DEFAULT_NVIDB_HOME}"
NVIDB="${NVIDB_BIN:-$DEFAULT_NVIDB}"
INTERVAL="${NVIDB_QUEUE_INTERVAL:-__INTERVAL__}"
MANAGER=__MANAGER__
KEEPER_TOKEN=__KEEPER_TOKEN__
UNIT=__SYSTEMD_UNIT__

PIDFILE="$NVIDB_HOME/queue-keeper.pid"
TOKENFILE="$NVIDB_HOME/queue-keeper.token"
LOCKDIR="$NVIDB_HOME/queue-keeper.lock"
LOCKOWNER="$LOCKDIR/owner"
SCRIPT="$NVIDB_HOME/queue-keeper.sh"
SESSION="$NVIDB_HOME/queue-keeper.session"
LOG="$NVIDB_HOME/queue-keeper.log"

read_pid() {
  [ -f "$PIDFILE" ] || return 1
  nvidb_pid=$(cat "$PIDFILE" 2>/dev/null) || return 1
  case "$nvidb_pid" in
    ""|*[!0-9]*) return 1 ;;
  esac
  printf "%s" "$nvidb_pid"
}

alive() {
  nvidb_pid=$(read_pid) || return 1
  kill -0 "$nvidb_pid" 2>/dev/null || return 1
  nvidb_command=$(ps -p "$nvidb_pid" -o command= 2>/dev/null) || return 1
  nvidb_token=$(cat "$TOKENFILE" 2>/dev/null) || nvidb_token=""
  if [ -n "$nvidb_token" ]; then
    case "$nvidb_command" in
      *"_loop $nvidb_token"*) return 0 ;;
      *) return 1 ;;
    esac
  fi
  # A keeper installed before identity tokens existed still uses this exact
  # script path. Recognizing it prevents an upgrade from starting a duplicate.
  case "$nvidb_command" in
    *"$SCRIPT _loop"*) return 0 ;;
    *) return 1 ;;
  esac
}

release_start_lock() {
  nvidb_lock_owner=$(cat "$LOCKOWNER" 2>/dev/null) || return 0
  [ "$nvidb_lock_owner" = "$$" ] || return 0
  rm -f "$LOCKOWNER"
  rmdir "$LOCKDIR" 2>/dev/null
}

wait_for_keeper() {
  nvidb_wait=0
  while [ "$nvidb_wait" -lt 5 ]; do
    alive && return 0
    sleep 1
    nvidb_wait=$((nvidb_wait + 1))
  done
  return 1
}

start_owner_alive() {
  nvidb_owner_pid="$1"
  case "$nvidb_owner_pid" in
    ""|*[!0-9]*) return 1 ;;
  esac
  [ "$nvidb_owner_pid" -gt 0 ] 2>/dev/null || return 1
  kill -0 "$nvidb_owner_pid" 2>/dev/null || return 1
  nvidb_owner_command=$(ps -p "$nvidb_owner_pid" -o command= 2>/dev/null) || return 1
  case "$nvidb_owner_command" in
    *"$SCRIPT start"*|*"$SCRIPT ensure"*|*"$SCRIPT") return 0 ;;
    *) return 1 ;;
  esac
}

clear_stale_start_lock() {
  # An empty directory can only be the tiny gap between mkdir and publishing
  # the owner, or debris from a process killed in that gap. rmdir is safe: it
  # fails atomically if the live starter publishes its owner first.
  if [ ! -f "$LOCKOWNER" ]; then
    rmdir "$LOCKDIR" 2>/dev/null
    return $?
  fi

  nvidb_lock_owner=$(cat "$LOCKOWNER" 2>/dev/null) || return 1
  start_owner_alive "$nvidb_lock_owner" && return 1

  # Moving the owner file is the cleanup claim. Only one contender can win it;
  # the others cannot remove a new owner's file after the directory is reused.
  nvidb_claim="$NVIDB_HOME/.queue-keeper.lock.stale.$$"
  rm -f "$nvidb_claim"
  mv "$LOCKOWNER" "$nvidb_claim" 2>/dev/null || return 1
  nvidb_claimed_owner=$(cat "$nvidb_claim" 2>/dev/null) || nvidb_claimed_owner=""

  if start_owner_alive "$nvidb_claimed_owner"; then
    if [ -d "$LOCKDIR" ] && [ ! -f "$LOCKOWNER" ]; then
      mv "$nvidb_claim" "$LOCKOWNER" 2>/dev/null
    else
      rm -f "$nvidb_claim"
    fi
    return 1
  fi

  if rmdir "$LOCKDIR" 2>/dev/null; then
    rm -f "$nvidb_claim"
    return 0
  fi
  if [ -d "$LOCKDIR" ] && [ ! -f "$LOCKOWNER" ]; then
    mv "$nvidb_claim" "$LOCKOWNER" 2>/dev/null
  else
    rm -f "$nvidb_claim"
  fi
  return 1
}

acquire_start_lock() {
  nvidb_attempt=0
  while [ "$nvidb_attempt" -lt 3 ]; do
    if mkdir "$LOCKDIR" 2>/dev/null; then
      if ! echo $$ > "$LOCKOWNER"; then
        rmdir "$LOCKDIR" 2>/dev/null
        return 1
      fi
      return 0
    fi

    # Another caller may be between creating the lock and publishing its pid.
    # Wait for that keeper instead of deleting a live startup lock.
    wait_for_keeper && return 2

    # A live starter may legitimately need longer than the readiness window.
    # Only a dead or malformed owner can be claimed and removed.
    clear_stale_start_lock || return 1
    nvidb_attempt=$((nvidb_attempt + 1))
  done
  return 1
}

systemd_action() {
  case "$1" in
    start)
      systemctl --user enable --now "$UNIT" || return 1
      systemctl --user is-active --quiet "$UNIT"
      ;;
    ensure)
      systemctl --user start "$UNIT" || return 1
      systemctl --user is-active --quiet "$UNIT"
      ;;
    stop)
      systemctl --user disable --now "$UNIT"
      ;;
    restart)
      systemctl --user enable "$UNIT" || return 1
      systemctl --user restart "$UNIT"
      ;;
    status)
      if systemctl --user is-active --quiet "$UNIT"; then
        nvidb_pid=$(systemctl --user show -p MainPID --value "$UNIT" 2>/dev/null)
        echo "running under systemd (pid ${nvidb_pid:-unknown})"
      else
        echo "stopped (systemd user service)"
        return 1
      fi
      ;;
    logs)
      journalctl --user -u "$UNIT" -n "${2:-50}" --no-pager
      ;;
    *)
      return 2
      ;;
  esac
}

action="${1:-start}"
if [ "$MANAGER" = "systemd" ]; then
  systemd_action "$action" "${2:-}" && exit 0
  rc=$?
  [ "$rc" -eq 2 ] && echo "usage: $0 {start|ensure|stop|restart|status|logs [LINES]}" >&2
  exit "$rc"
fi

case "$action" in
  start|ensure)
    alive && exit 0
    mkdir -p "$NVIDB_HOME" || exit 1
    acquire_start_lock
    lock_rc=$?
    [ "$lock_rc" -eq 2 ] && exit 0
    if [ "$lock_rc" -ne 0 ]; then
      echo "could not acquire the keeper startup lock at $LOCKDIR" >&2
      exit 1
    fi

    echo "$KEEPER_TOKEN" > "$TOKENFILE"
    if command -v setsid >/dev/null 2>&1; then
      # Its own session lets the keeper outlive the SSH channel and gives stop
      # a process group containing both the keeper and its daemon.
      echo 1 > "$SESSION"
      setsid "$0" _loop "$KEEPER_TOKEN" >> "$LOG" 2>&1 < /dev/null &
    else
      # macOS has no setsid. nohup detaches from the terminal, but the keeper
      # still shares a process group with its caller and must be stopped by pid.
      echo 0 > "$SESSION"
      nohup "$0" _loop "$KEEPER_TOKEN" >> "$LOG" 2>&1 < /dev/null &
    fi
    starter_pid=$!

    if wait_for_keeper; then
      release_start_lock
      exit 0
    fi

    kill -TERM "$starter_pid" 2>/dev/null
    rm -f "$PIDFILE" "$TOKENFILE"
    release_start_lock
    echo "keeper did not become ready; see $LOG" >&2
    exit 1
    ;;
  _loop)
    loop_token="${2:-}"
    if [ -z "$loop_token" ] || [ "$loop_token" != "$KEEPER_TOKEN" ]; then
      echo "[keeper] refusing an internal start with an invalid token" >&2
      exit 2
    fi
    echo "$loop_token" > "$TOKENFILE"
    echo $$ > "$PIDFILE"
    daemon_pid=""

    nvidb_shutdown() {
      if [ "$(cat "$PIDFILE" 2>/dev/null)" = "$$" ]; then
        rm -f "$PIDFILE" "$TOKENFILE"
      fi
      [ -n "$daemon_pid" ] && kill -TERM "$daemon_pid" 2>/dev/null
      echo "[keeper] $(date) stopped"
      exit 143
    }
    trap nvidb_shutdown TERM INT
    echo "[keeper] $(date) up, interval=${INTERVAL}s, nvidb=$NVIDB"
    while : ; do
      # Waiting on a background child lets the signal trap run immediately.
      "$NVIDB" queue daemon --interval "$INTERVAL" &
      daemon_pid=$!
      wait "$daemon_pid"
      rc=$?
      daemon_pid=""
      echo "[keeper] $(date) daemon exited ($rc), restarting in 5s"
      sleep 5
    done
    ;;
  stop)
    if alive; then
      pid=$(read_pid)
      if [ "$(cat "$SESSION" 2>/dev/null)" = "1" ]; then
        kill -TERM "-$pid" 2>/dev/null
      else
        kill -TERM "$pid" 2>/dev/null
        pkill -TERM -P "$pid" 2>/dev/null
      fi
    fi
    rm -f "$PIDFILE" "$TOKENFILE" "$LOCKOWNER"
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
      echo "running (pid $(read_pid))"
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
