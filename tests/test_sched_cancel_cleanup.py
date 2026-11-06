"""Cancellation must follow known children after their wrapper exits."""
import os
import signal
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nvidb.sched.executor import JobExecutor, build_run_script
from nvidb.sched.transport import LocalTransport
from nvidb.sched.termination import TERMINATION_SCRIPT


def _wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    pytest.fail('cancellation did not clean up its child process')


def _working(pid):
    result = subprocess.run(
        ['ps', '-o', 'stat=', '-p', str(pid)], text=True, capture_output=True,
    )
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith('Z')


@pytest.mark.parametrize('known_pid', [True, False], ids=['confirmed', 'unconfirmed'])
@pytest.mark.parametrize('private_group', [True, False], ids=['session', 'children'])
def test_cancel_kills_a_term_ignoring_child_after_its_wrapper_exits(
    tmp_path, monkeypatch, known_pid, private_group,
):
    # A detached job can inherit this environment value after its launch lock
    # has closed. It must not be mistaken for an open descriptor in the watcher.
    monkeypatch.setenv('NVIDB_LAUNCH_LOCK_FD', '3')
    run_dir = tmp_path / 'job with spaces'
    run_dir.mkdir()
    (run_dir / 'session').write_text('1' if private_group else '0')
    script = run_dir / 'run.sh'
    script.write_text(build_run_script(
        job_id=1, job_name='cancel-regression', run_dir=str(run_dir),
        workdir=None, command=(
            '(trap "" TERM; : > "$NVIDB_JOB_DIR/child.ready"; exec sleep 60) &\n'
            'echo $! > "$NVIDB_JOB_DIR/child.pid"\nwait\n'
        ),
    ))
    # A private session also makes test cleanup safe on macOS, where the real
    # launcher uses nohup. With pgid=None we exercise that child-only path.
    wrapper = subprocess.Popen(
        ['bash', str(script)], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    child = None
    try:
        _wait_for(lambda: (run_dir / 'child.pid').exists() and (run_dir / 'child.ready').exists())
        child = int((run_dir / 'child.pid').read_text())
        executor = JobExecutor(LocalTransport('cancel-test'))
        assert executor.terminate(
            run_dir=str(run_dir), pid=wrapper.pid if known_pid else None,
            pgid=wrapper.pid if private_group and known_pid else None, grace=2,
        )
        _wait_for(lambda: wrapper.poll() is not None)
        assert wrapper.returncode == 143
        assert _working(child)  # TERM alone cannot stop this child.
        _wait_for(lambda: not _working(child))
    finally:
        try:
            os.killpg(wrapper.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        wrapper.wait(timeout=5)


def test_escalation_refuses_a_recycled_process_group(monkeypatch):
    namespace = {'__name__': 'termination_test'}
    exec(TERMINATION_SCRIPT, namespace)
    monkeypatch.setitem(namespace, 'process_table', lambda: {
        123: {'group': 123, 'started': 'new process'},
    })
    monkeypatch.setitem(namespace, 'identity', lambda pid, row: (row['group'], row['started']))
    signalled = []
    monkeypatch.setitem(namespace, 'send', lambda *args: signalled.append(args))
    with pytest.raises(RuntimeError, match='identity is no longer verifiable'):
        namespace['escalate']({123: (123, 'original process')}, 123, 0)
    assert signalled == []


def test_cancellation_stops_when_process_listing_fails(tmp_path, monkeypatch):
    namespace = {'__name__': 'termination_test'}
    exec(TERMINATION_SCRIPT, namespace)
    def unavailable():
        raise subprocess.CalledProcessError(1, 'ps')
    monkeypatch.setitem(namespace, 'process_table', unavailable)
    signalled = []
    monkeypatch.setitem(namespace, 'send', lambda *args: signalled.append(args))
    with pytest.raises(subprocess.CalledProcessError):
        namespace['terminate'](tmp_path, 123, 123, 0, False, False)
    assert signalled == []


CUDA_CHILD = r'''
import ctypes, os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
cuda = ctypes.CDLL('libcuda.so.1')
def call(name, *args):
    result = getattr(cuda, name)(*args)
    if result != 0:
        raise RuntimeError('%s failed: CUDA error %d' % (name, result))
call('cuInit', 0)
device = ctypes.c_int()
call('cuDeviceGet', ctypes.byref(device), 0)
context = ctypes.c_void_p()
call('cuCtxCreate_v2', ctypes.byref(context), 0, device)
memory = ctypes.c_uint64()
call('cuMemAlloc_v2', ctypes.byref(memory), ctypes.c_size_t(64 * 1024 * 1024))
Path(os.environ['NVIDB_JOB_DIR'], 'cuda.ready').write_text(str(os.getpid()))
time.sleep(60)
'''


@pytest.mark.skipif(os.environ.get('NVIDB_TEST_CUDA') != '1', reason='requires an explicitly reserved CUDA GPU')
def test_cancel_releases_real_cuda_memory_after_the_wrapper_exits(tmp_path):
    import pynvml
    visible = [int(index) for index in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    assert len(visible) == 1
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(visible[0])
    executor = JobExecutor(LocalTransport('cuda-cancel-test'), job_root=str(tmp_path / 'jobs'))
    launched = executor.launch(
        job_id=1, job_name='cuda-cancel', gpu_ids=visible,
        command=shlex.join([sys.executable, '-c', CUDA_CHILD]) + ' &\nwait',
    )
    run_dir = Path(launched.run_dir)
    try:
        assert launched.pgid is not None
        _wait_for(lambda: (run_dir / 'cuda.ready').exists(), timeout=25)
        child = int((run_dir / 'cuda.ready').read_text())
        def allocations():
            return {process.pid: process.usedGpuMemory for process in
                    pynvml.nvmlDeviceGetComputeRunningProcesses(handle)}
        _wait_for(lambda: allocations().get(child, 0) >= 64 * 1024 * 1024)
        assert executor.terminate(run_dir=str(run_dir), pid=launched.pid, pgid=launched.pgid, grace=2)
        _wait_for(lambda: (run_dir / 'exit_code').exists())
        assert (run_dir / 'exit_code').read_text().split()[0] == '143'
        assert _working(child)
        _wait_for(lambda: not _working(child))
        _wait_for(lambda: child not in allocations(), timeout=20)
    finally:
        try:
            os.killpg(launched.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        pynvml.nvmlShutdown()
