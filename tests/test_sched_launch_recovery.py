"""Launch intent survives controller crashes and lost SSH acknowledgements."""
import pytest

from fake_cluster import make_scheduler, two_node_cluster
from nvidb.sched import db as dbm
from nvidb.sched.executor import LaunchResult
from nvidb.sched.transport import TransportError


@pytest.fixture
def queue(tmp_path):
    cluster = two_node_cluster()
    conn = dbm.open_db(tmp_path / 'queue.db')
    scheduler = make_scheduler(conn, cluster)
    scheduler.tick(force=True)
    yield scheduler, cluster
    scheduler.close()
    conn.close()


def lose_launch_reply(cluster, monkeypatch, *, finish=False, crash=False):
    executor = cluster.backends['big-node'].executor
    launch = executor.launch

    def uncertain(**kwargs):
        result = launch(**kwargs)
        if finish:
            cluster['big-node'].finish_job(kwargs['job_id'])
        else:
            cluster['big-node'].job_allocates(kwargs['job_id'], 0, 70000)
        if crash:
            raise KeyboardInterrupt('controller died after remote execution')
        raise TransportError('remote launch succeeded; acknowledgement lost')

    monkeypatch.setattr(executor, 'launch', uncertain)
    return launch


def test_lost_acknowledgement_keeps_placement_and_recovers_without_a_second_copy(queue, monkeypatch):
    scheduler, cluster = queue
    job_id = scheduler.submit('train', vram='1G', max_retries=2)
    lose_launch_reply(cluster, monkeypatch)
    scheduler.tick(force=True)
    uncertain = dbm.get_job(scheduler.conn, job_id)
    assert uncertain.state == 'running'
    assert uncertain.node == 'big-node'
    assert uncertain.run_dir == '/fake/big-node/jobs/1'
    assert uncertain.attempt == 1
    assert uncertain.remote_pid is None

    scheduler.tick(force=True)
    recovered = dbm.get_job(scheduler.conn, job_id)
    assert recovered.node == 'big-node'
    assert recovered.remote_pid == cluster['big-node'].jobs[job_id]['pid']
    assert recovered.attempt == 1
    assert job_id not in cluster['small-node'].jobs
    scheduler.tick(force=True)
    starts = [event for event in dbm.list_events(scheduler.conn, job_id=job_id)
              if event['kind'] == 'job_started']
    assert len(starts) == 1
    assert starts[0]['data']['recovered']
    assert scheduler.cancel(job_id)
    assert not cluster['big-node'].jobs[job_id]['alive']


@pytest.mark.parametrize('offline', [False, True])
def test_cancel_finds_an_unconfirmed_launch_by_attempt_directory(queue, monkeypatch, offline):
    scheduler, cluster = queue
    job_id = scheduler.submit('train', vram='1G')
    lose_launch_reply(cluster, monkeypatch)
    scheduler.tick(force=True)
    cluster['big-node'].online = not offline
    assert scheduler.cancel(job_id)
    assert dbm.get_job(scheduler.conn, job_id).state == 'cancelled'
    if offline:
        orphan = dbm.list_orphans(scheduler.conn)[0]
        assert orphan['pid'] is None
        assert orphan['run_dir'] == '/fake/big-node/jobs/1'
        cluster['big-node'].online = True
        scheduler.tick(force=True)
    assert not cluster['big-node'].jobs[job_id]['alive']
    assert not dbm.list_orphans(scheduler.conn)


@pytest.mark.parametrize('crash', [False, True])
def test_completed_uncertain_attempt_is_settled_without_replaying(queue, monkeypatch, crash):
    scheduler, cluster = queue
    job_id = scheduler.submit('quick command', vram='1G')
    lose_launch_reply(cluster, monkeypatch, finish=True, crash=crash)
    if crash:
        with pytest.raises(KeyboardInterrupt):
            scheduler.tick(force=True)
    else:
        scheduler.tick(force=True)
    original_pid = cluster['big-node'].jobs[job_id]['pid']
    scheduler.tick(force=True)
    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == 'completed'
    assert job.exit_code == 0
    assert job.attempt == 1
    assert job.remote_pid == original_pid
    assert job_id not in cluster['small-node'].jobs


def test_controller_crash_before_send_recovers_reserved_attempt_on_same_node(queue, monkeypatch):
    scheduler, cluster = queue
    job_id = scheduler.submit('train', vram='1G')
    executor = cluster.backends['big-node'].executor
    original = executor.launch

    def crash_before_send(**kwargs):
        reserved = dbm.get_job(scheduler.conn, job_id)
        assert reserved.state == 'running'
        assert reserved.node == 'big-node'
        assert reserved.run_dir == kwargs['run_dir']
        raise KeyboardInterrupt('controller died before sending launch')

    monkeypatch.setattr(executor, 'launch', crash_before_send)
    with pytest.raises(KeyboardInterrupt):
        scheduler.tick(force=True)
    monkeypatch.setattr(executor, 'launch', original)
    recovered_scheduler = make_scheduler(scheduler.conn, cluster)
    try:
        recovered_scheduler.tick(force=True)
        job = dbm.get_job(scheduler.conn, job_id)
        assert job.state == 'running'
        assert job.remote_pid is not None
        assert job.node == 'big-node'
        assert job.attempt == 1
        assert job_id not in cluster['small-node'].jobs
    finally:
        recovered_scheduler.close()


def test_claim_without_pid_keeps_reservation_instead_of_declaring_loss(queue, monkeypatch):
    scheduler, cluster = queue
    job_id = scheduler.submit('train', vram='1G', max_retries=5)
    sent = []

    def claim_without_pid(**kwargs):
        sent.append((kwargs['node_name'], kwargs['attempt'], kwargs['run_dir']))
        return LaunchResult(None, None, kwargs['run_dir'], adopted=True)

    monkeypatch.setattr(cluster.backends['big-node'].executor, 'launch', claim_without_pid)
    for _ in range(3):
        scheduler.tick(force=True)
    job = dbm.get_job(scheduler.conn, job_id)
    assert job.state == 'running'
    assert job.remote_pid is None
    assert job.attempt == 1
    assert len(set(sent)) == 1
    assert 'waiting' in job.last_error
    assert job_id not in cluster['small-node'].jobs


def test_uncertain_start_consumes_budget_for_remaining_jobs_in_the_same_tick(queue, monkeypatch):
    scheduler, cluster = queue
    first = scheduler.submit('first', vram='1G')
    second = scheduler.submit('second', vram='1G')
    lose_launch_reply(cluster, monkeypatch)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, first).node == 'big-node'
    assert dbm.get_job(scheduler.conn, second).node == 'small-node'
    assert second not in cluster['big-node'].jobs


def test_recovery_preserves_the_original_exclusive_allocation(queue, monkeypatch):
    scheduler, cluster = queue
    first = scheduler.submit('first', vram='1G', node='big-node')
    executor = cluster.backends['big-node'].executor
    original = executor.launch

    def crash_before_send(**kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(executor, 'launch', crash_before_send)
    with pytest.raises(KeyboardInterrupt):
        scheduler.tick(force=True)
    scheduler.settings['gpu_mode'] = 'shared'
    second = scheduler.submit('second', vram='1G', node='big-node')
    monkeypatch.setattr(executor, 'launch', original)
    scheduler.tick(force=True)
    assert dbm.get_job(scheduler.conn, first).allocation_mode == 'exclusive'
    assert dbm.get_job(scheduler.conn, second).state == 'pending'
