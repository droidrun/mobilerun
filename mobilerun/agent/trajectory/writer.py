import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import aiofiles
from PIL import Image

from aiofiles import ospath

logger = logging.getLogger("droidrun")


def make_serializable(obj):
    """Recursively make objects JSON serializable."""
    if hasattr(obj, "__class__") and obj.__class__.__name__ == "ChatMessage":
        if hasattr(obj, "content") and obj.content is not None:
            return {"role": obj.role.value, "content": obj.content}
        elif hasattr(obj, "blocks") and obj.blocks:
            text_content = ""
            for block in obj.blocks:
                if hasattr(block, "text"):
                    text_content += block.text
            return {"role": obj.role.value, "content": text_content}
        else:
            return str(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        result = {}
        for k, v in obj.__dict__.items():
            if not k.startswith("_"):
                try:
                    result[k] = make_serializable(v)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Failed to serialize attribute {k}: {e}")
                    result[k] = str(v)
        return result
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


@dataclass(frozen=True)
class WriteJob(ABC):
    """Immutable write operation.

    Jobs capture a snapshot of data at creation time, preventing race
    conditions from concurrent trajectory mutations during async writes.
    """

    trajectory_id: str
    stage: str

    @abstractmethod
    async def execute(self) -> None:
        """Execute the write operation asynchronously."""
        pass


@dataclass(frozen=True)
class EventsWriteJob(WriteJob):
    """Writes trajectory.json."""

    target_path: Path
    serialized_events: str

    async def execute(self) -> None:
        async with aiofiles.open(self.target_path, "w") as f:
            await f.write(self.serialized_events)


@dataclass(frozen=True)
class MacroWriteJob(WriteJob):
    """Writes macro.json."""

    target_path: Path
    serialized_macro: str

    async def execute(self) -> None:
        async with aiofiles.open(self.target_path, "w") as f:
            await f.write(self.serialized_macro)


@dataclass(frozen=True)
class ScreenshotWriteJob(WriteJob):
    """Writes individual screenshot PNG."""

    target_path: Path
    screenshot_bytes: bytes

    async def execute(self) -> None:
        async with aiofiles.open(self.target_path, "wb") as f:
            await f.write(self.screenshot_bytes)


@dataclass(frozen=True)
class GifWriteJob(WriteJob):
    """Creates animated GIF from screenshots on disk."""

    target_path: Path
    screenshots_folder: Path
    screenshot_count: int
    duration: int = 1000

    async def execute(self) -> None:
        def _create_gif():
            images = []

            for idx in range(self.screenshot_count):
                screenshot_path = self.screenshots_folder / f"{idx:04d}.png"
                if not screenshot_path.exists():
                    logger.warning(f"Screenshot {idx:04d}.png not found, skipping")
                    continue

                try:
                    img = Image.open(screenshot_path)
                    images.append(img.copy())
                    img.close()
                except Exception as e:
                    logger.warning(f"Failed to load screenshot {idx}: {e}")
                    continue

            if images:
                try:
                    images[0].save(
                        str(self.target_path),
                        save_all=True,
                        append_images=images[1:],
                        duration=self.duration,
                        loop=0,
                    )
                finally:
                    for img in images:
                        try:
                            img.close()
                        except Exception:
                            pass

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _create_gif)


@dataclass(frozen=True)
class UIStateWriteJob(WriteJob):
    """Writes UI state JSON."""

    target_path: Path
    serialized_state: str

    async def execute(self) -> None:
        if not (await ospath.exists(self.target_path)):
            async with aiofiles.open(self.target_path, "w", encoding="utf-8") as f:
                await f.write(self.serialized_state)


class WriterWorker:
    """Async worker that processes write jobs from a queue.

    Jobs are executed sequentially (FIFO order). Errors are logged but
    don't stop the worker - it continues processing remaining jobs.
    """

    def __init__(self, max_queue_size: int = 300):
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.worker_task = None
        self.running = False
        self._write_count = 0
        self._error_count = 0

    async def start(self) -> None:
        """Start the worker task."""
        if self.running:
            return

        self.running = True
        self.worker_task = asyncio.create_task(self._work_loop())

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop worker and wait for pending writes to complete.

        Drains the queue before canceling. Any jobs still pending after
        timeout are dropped with a warning.

        Args:
            timeout: Maximum seconds to wait for queue to drain
        """
        if not self.running:
            return

        try:
            await asyncio.wait_for(self.queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                f"Writer timeout after {timeout}s, {self.queue.qsize()} jobs remaining"
            )

        self.running = False

        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def _work_loop(self) -> None:
        while self.running:
            try:
                job = await self.queue.get()

                try:
                    await job.execute()
                    self._write_count += 1
                except Exception as e:
                    self._error_count += 1
                    logger.error(f"Write failed: {e}", exc_info=True)
                finally:
                    self.queue.task_done()

            except asyncio.CancelledError:
                break

    def submit(self, job: WriteJob) -> bool:
        try:
            self.queue.put_nowait(job)
            return True
        except asyncio.QueueFull:
            logger.warning(f"Queue full, dropping {job.__class__.__name__}")
            return False


class TrajectoryWriter:
    """Background writer that persists trajectory data without blocking.

    Manages a single async worker that processes write jobs from a queue.
    All writes happen in the background - agent never blocks on I/O.

    Usage:
        writer = TrajectoryWriter(queue_size=300)
        await writer.start()

        # Non-blocking writes
        writer.write(trajectory, stage="step_1")

        # Wait for completion
        writer.write_final(trajectory)
        await writer.stop()
    """

    def __init__(self, queue_size: int = 300):
        self.worker = WriterWorker(max_queue_size=queue_size)
        self._started = False

    async def start(self) -> None:
        """Start the background worker."""
        await self.worker.start()
        self._started = True

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the background worker and wait for pending writes."""
        await self.worker.stop(timeout=timeout)

    def write(self, trajectory, stage: str) -> None:
        """Write trajectory data to disk in background (non-blocking).

        Creates a snapshot of trajectory data and queues write jobs.
        Returns immediately - actual writes happen asynchronously.

        Args:
            trajectory: Trajectory instance with data to persist
            stage: Label for this save point (e.g. "init", "step_1", "final")
        """
        if not self._started:
            logger.warning("TrajectoryWriter not started, skipping save")
            return

        trajectory_id = trajectory.trajectory_folder.name

        events_snapshot = list(trajectory.events)
        macro_snapshot = list(trajectory.macro) if trajectory.macro else []
        screenshot_queue_snapshot = list(trajectory.screenshot_queue)
        ui_states_snapshot = list(trajectory.ui_states)

        jobs = []

        events_job = self._create_events_job(
            events_snapshot, trajectory, trajectory_id, stage
        )
        if events_job:
            jobs.append(events_job)

        macro_job = self._create_macro_job(
            macro_snapshot, trajectory, trajectory_id, stage
        )
        if macro_job:
            jobs.append(macro_job)

        screenshot_jobs = self._create_screenshot_jobs(
            screenshot_queue_snapshot, trajectory, trajectory_id, stage
        )
        jobs.extend(screenshot_jobs)

        ui_jobs = self._create_ui_state_jobs(
            ui_states_snapshot, trajectory, trajectory_id, stage
        )
        jobs.extend(ui_jobs)

        for job in jobs:
            self.worker.submit(job)

        if screenshot_queue_snapshot:
            trajectory.screenshot_queue.clear()
            logger.debug(
                f"Cleared {len(screenshot_queue_snapshot)} screenshots from queue"
            )

    def write_final(self, trajectory, trajectory_gifs) -> None:
        """Write final trajectory data including GIF creation.

        Args:
            trajectory: Trajectory instance to finalize
        """
        self.write(trajectory, stage="final")
        # GIF is only created at finalize (all screenshots available)
        if trajectory_gifs is True:
            trajectory_id = trajectory.trajectory_folder.name
            gif_job = self._create_gif_job(trajectory, trajectory_id, "final")
            if gif_job:
                self.worker.submit(gif_job)

    def _create_events_job(
        self, events_snapshot, trajectory, trajectory_id, stage
    ) -> Optional[EventsWriteJob]:
        serialized_events = []

        for event in events_snapshot:
            event_dict = {"type": event.__class__.__name__}

            for k, v in event.__dict__.items():
                if not k.startswith("_"):
                    try:
                        event_dict[k] = make_serializable(v)
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to serialize {k}: {e}")
                        event_dict[k] = str(v)

            if hasattr(event, "tokens") and "tokens" not in event_dict:
                event_dict["tokens"] = make_serializable(event.tokens)

            serialized_events.append(event_dict)

        return EventsWriteJob(
            trajectory_id=trajectory_id,
            stage=stage,
            target_path=trajectory.trajectory_folder / "trajectory.json",
            serialized_events=json.dumps(serialized_events, indent=2),
        )

    def _create_macro_job(
        self, macro_snapshot, trajectory, trajectory_id, stage
    ) -> Optional[MacroWriteJob]:
        if not macro_snapshot:
            return None

        macro_data = {
            "version": "1.0",
            "description": trajectory.goal,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "total_actions": len(macro_snapshot),
            "actions": macro_snapshot,  # already list[dict] from RecordingDriver.log
        }

        return MacroWriteJob(
            trajectory_id=trajectory_id,
            stage=stage,
            target_path=trajectory.trajectory_folder / "macro.json",
            serialized_macro=json.dumps(macro_data, indent=2),
        )

    def _create_screenshot_jobs(
        self, screenshot_queue_snapshot, trajectory, trajectory_id, stage
    ) -> List[ScreenshotWriteJob]:
        jobs = []
        screenshots_folder = trajectory.trajectory_folder / "screenshots"
        screenshots_folder.mkdir(exist_ok=True)

        start_idx = trajectory.screenshot_count - len(screenshot_queue_snapshot)

        for offset, screenshot_bytes in enumerate(screenshot_queue_snapshot):
            screenshot_idx = start_idx + offset
            screenshot_path = screenshots_folder / f"{screenshot_idx:04d}.png"
            jobs.append(
                ScreenshotWriteJob(
                    trajectory_id=trajectory_id,
                    stage=stage,
                    target_path=screenshot_path,
                    screenshot_bytes=screenshot_bytes,
                )
            )

        return jobs

    def _create_gif_job(
        self, trajectory, trajectory_id, stage
    ) -> Optional[GifWriteJob]:
        if trajectory.screenshot_count == 0:
            return None

        screenshots_folder = trajectory.trajectory_folder / "screenshots"
        screenshots_folder.mkdir(exist_ok=True)

        return GifWriteJob(
            trajectory_id=trajectory_id,
            stage=stage,
            target_path=screenshots_folder / "trajectory.gif",
            screenshots_folder=screenshots_folder,
            screenshot_count=trajectory.screenshot_count,
        )

    def _create_ui_state_jobs(
        self, ui_states_snapshot, trajectory, trajectory_id, stage
    ) -> List[UIStateWriteJob]:
        jobs = []
        ui_states_folder = trajectory.trajectory_folder / "ui_states"
        ui_states_folder.mkdir(exist_ok=True)

        for idx, ui_state in enumerate(ui_states_snapshot):
            ui_state_path = ui_states_folder / f"{idx:04d}.json"
            jobs.append(
                UIStateWriteJob(
                    trajectory_id=trajectory_id,
                    stage=stage,
                    target_path=ui_state_path,
                    serialized_state=json.dumps(ui_state, ensure_ascii=False, indent=2),
                )
            )

        return jobs
