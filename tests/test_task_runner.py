import asyncio

from mobilerun.orchestration.models import TaskRequest, TaskResult
from mobilerun.orchestration.runner import TaskRunner


class FakeQueue:
    """In-memory queue double. Logs mark_* calls; dequeue drains a preloaded list."""

    def __init__(self, requests):
        self._requests = list(requests)
        self.calls = []  # ordered log of (op, task_id) tuples
        self.results = {}  # task_id -> TaskResult passed to mark_completed/mark_failed

    async def dequeue(self):
        if not self._requests:
            # Mirror the real queue idling on an empty/not-ready queue: block
            # forever. Callers stop the loop via TaskRunner.stop() or by
            # cancelling the asyncio.Task wrapping run_forever().
            await asyncio.Event().wait()
        request = self._requests.pop(0)
        self.calls.append(("dequeue", request.id))
        return request

    def mark_running(self, task_id):
        self.calls.append(("mark_running", task_id))

    def mark_completed(self, task_id, result):
        self.calls.append(("mark_completed", task_id))
        self.results[task_id] = result

    def mark_failed(self, task_id, result):
        self.calls.append(("mark_failed", task_id))
        self.results[task_id] = result


class FakeInvoker:
    """Returns preloaded TaskResults in order, or raises a preloaded exception.

    results: list of TaskResult returned per successive run_task call.
    raise_on: dict {call_index -> Exception} -- raise instead of returning.
    on_call: optional callable(index) invoked at the start of each run_task,
             used by tests to trigger stop() mid-task.
    """

    def __init__(self, results=None, raise_on=None, on_call=None):
        self._results = list(results or [])
        self._raise_on = raise_on or {}
        self._on_call = on_call
        self.received_requests = []
        self.received_event_callbacks = []

    async def run_task(self, request, event_callback=None):
        index = len(self.received_requests)
        self.received_requests.append(request)
        self.received_event_callbacks.append(event_callback)
        if self._on_call is not None:
            self._on_call(index)
        if index in self._raise_on:
            raise self._raise_on[index]
        return self._results[index]


def _request(goal="do a thing"):
    return TaskRequest(goal=goal)


def _result(task_id, success=True, reason="ok", steps=1):
    return TaskResult(task_id=task_id, success=success, reason=reason, steps=steps)


def test_run_once_happy_path_transitions_and_call_order():
    request = _request()
    invoker_result = _result(request.id, success=True)
    queue = FakeQueue([request])
    invoker = FakeInvoker(results=[invoker_result])
    runner = TaskRunner(queue, invoker)

    returned = asyncio.run(runner.run_once())

    assert queue.calls == [
        ("dequeue", request.id),
        ("mark_running", request.id),
        ("mark_completed", request.id),
    ]
    assert returned is invoker_result
    assert invoker.received_requests == [request]


def test_run_once_forwards_event_callback():
    request = _request()
    queue = FakeQueue([request])
    invoker = FakeInvoker(results=[_result(request.id)])
    sentinel = object()
    runner = TaskRunner(queue, invoker, event_callback=sentinel)

    asyncio.run(runner.run_once())

    assert invoker.received_event_callbacks == [sentinel]


def test_run_once_invoker_returns_failure_marks_failed():
    request = _request()
    failed = _result(request.id, success=False, reason="agent gave up")
    queue = FakeQueue([request])
    invoker = FakeInvoker(results=[failed])
    runner = TaskRunner(queue, invoker)

    returned = asyncio.run(runner.run_once())

    assert ("mark_failed", request.id) in queue.calls
    assert ("mark_completed", request.id) not in queue.calls
    assert queue.results[request.id] is failed
    assert returned is failed


def test_run_once_invoker_raises_converts_to_synthetic_failed_result():
    request = _request()
    queue = FakeQueue([request])
    invoker = FakeInvoker(raise_on={0: RuntimeError("boom")})
    runner = TaskRunner(queue, invoker)

    returned = asyncio.run(runner.run_once())

    # Did not raise; converted to a synthetic failed result.
    assert returned.success is False
    assert returned.task_id == request.id
    assert "RuntimeError" in returned.reason
    assert "boom" in returned.reason
    assert returned.error is not None and "boom" in returned.error
    # mark_running happened before the failure; mark_failed, never mark_completed.
    assert queue.calls == [
        ("dequeue", request.id),
        ("mark_running", request.id),
        ("mark_failed", request.id),
    ]


def test_run_forever_processes_multiple_tasks_sequentially():
    requests = [_request(f"goal {i}") for i in range(3)]
    queue = FakeQueue(requests)
    invoker = FakeInvoker(results=[_result(r.id) for r in requests])
    runner = TaskRunner(queue, invoker)

    async def scenario():
        task = asyncio.create_task(runner.run_forever())
        # Wait until all three completions land, then stop and cancel.
        while sum(1 for c in queue.calls if c[0] == "mark_completed") < 3:
            await asyncio.sleep(0)
        runner.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    # Strict alternation: dequeue -> mark_running -> mark_completed, per task,
    # never two mark_running without an intervening completion/failure.
    expected = []
    for r in requests:
        expected += [
            ("dequeue", r.id),
            ("mark_running", r.id),
            ("mark_completed", r.id),
        ]
    assert queue.calls == expected


def test_run_forever_continues_after_failing_invoker():
    req_fail, req_ok = _request("fails"), _request("ok")
    queue = FakeQueue([req_fail, req_ok])
    invoker = FakeInvoker(
        # index 0 raises (see raise_on), so its result slot is never read.
        results=[None, _result(req_ok.id)],
        raise_on={0: RuntimeError("first task explodes")},
    )
    runner = TaskRunner(queue, invoker)

    async def scenario():
        task = asyncio.create_task(runner.run_forever())
        while ("mark_completed", req_ok.id) not in queue.calls:
            await asyncio.sleep(0)
        runner.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert ("mark_failed", req_fail.id) in queue.calls
    assert ("mark_completed", req_ok.id) in queue.calls


def test_stop_after_current_task_exits_loop_before_next_dequeue():
    req1, req2 = _request("first"), _request("second")
    queue = FakeQueue([req1, req2])

    def _stop_during_first(index):
        if index == 0:
            runner.stop()

    invoker = FakeInvoker(
        results=[_result(req1.id), _result(req2.id)],
        on_call=_stop_during_first,
    )
    runner = TaskRunner(queue, invoker)

    asyncio.run(asyncio.wait_for(runner.run_forever(), timeout=1.0))

    # First task fully processed; second never dequeued.
    assert ("mark_completed", req1.id) in queue.calls
    assert ("dequeue", req2.id) not in queue.calls
    assert ("mark_running", req2.id) not in queue.calls


def test_stop_called_while_idle_does_not_hang_forever_run():
    # Empty queue -> dequeue() blocks forever. stop() alone does NOT unblock it;
    # this documents the deliberate tradeoff (cancel the task for immediate stop).
    queue = FakeQueue([])
    invoker = FakeInvoker()
    runner = TaskRunner(queue, invoker)

    async def scenario():
        task = asyncio.create_task(runner.run_forever())
        runner.stop()  # flag set, but the loop is parked inside dequeue()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
            raise AssertionError("run_forever should still be blocked in dequeue()")
        except asyncio.TimeoutError:
            pass
        assert not task.done()
        # Cancellation is the intended escape hatch for an idle loop.
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled()

    asyncio.run(scenario())


def test_run_forever_cancellation_propagates_cleanly():
    # A blocked run_forever() cancels cleanly -- CancelledError is not swallowed
    # by run_once's `except Exception` (it subclasses BaseException).
    queue = FakeQueue([])
    invoker = FakeInvoker()
    runner = TaskRunner(queue, invoker)

    async def scenario():
        task = asyncio.create_task(runner.run_forever())
        await asyncio.sleep(0)  # let it reach the blocked dequeue()
        task.cancel()
        with_error = None
        try:
            await task
        except asyncio.CancelledError as exc:
            with_error = exc
        assert with_error is not None
        assert task.cancelled()

    asyncio.run(scenario())
