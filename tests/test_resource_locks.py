import logging
import multiprocessing
import queue
import tempfile
import time
import unittest
from pathlib import Path

from video_analyzer.resource_locks import FileResourceLease, resource_lock_settings


def hold_resource(lock_dir: str, release: multiprocessing.Event, messages: multiprocessing.Queue) -> None:
    logger = logging.getLogger("test_resource_locks.hold")
    with FileResourceLease(
        resource="asr",
        limit=1,
        lock_dir=Path(lock_dir),
        owner="holder",
        poll_seconds=0.05,
        log_interval_seconds=0.1,
        logger=logger,
    ):
        messages.put("holder-acquired")
        release.wait(5)


def wait_for_resource(lock_dir: str, messages: multiprocessing.Queue) -> None:
    logger = logging.getLogger("test_resource_locks.wait")
    started = time.monotonic()
    with FileResourceLease(
        resource="asr",
        limit=1,
        lock_dir=Path(lock_dir),
        owner="waiter",
        poll_seconds=0.05,
        log_interval_seconds=0.1,
        logger=logger,
    ):
        messages.put(("waiter-acquired", round(time.monotonic() - started, 2)))


class ResourceLockTests(unittest.TestCase):
    def test_file_resource_lease_blocks_second_process_until_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            release = multiprocessing.Event()
            messages: multiprocessing.Queue = multiprocessing.Queue()
            holder = multiprocessing.Process(target=hold_resource, args=(tmp, release, messages))
            waiter = multiprocessing.Process(target=wait_for_resource, args=(tmp, messages))
            holder.start()
            try:
                self.assertEqual(messages.get(timeout=2), "holder-acquired")
                waiter.start()
                with self.assertRaises(queue.Empty):
                    messages.get(timeout=0.2)
                release.set()
                acquired, waited = messages.get(timeout=2)
                self.assertEqual(acquired, "waiter-acquired")
                self.assertGreaterEqual(waited, 0.2)
            finally:
                release.set()
                holder.join(timeout=2)
                waiter.join(timeout=2)
                if holder.is_alive():
                    holder.terminate()
                if waiter.is_alive():
                    waiter.terminate()

    def test_resource_lock_settings_use_task_level_defaults(self):
        settings = resource_lock_settings({"resource_locks": {"limits": {"asr": 3}}}, "ocr")

        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["limit"], 1)
        self.assertEqual(settings["lock_dir"], Path("tmp/video-link-status/resource-locks"))


if __name__ == "__main__":
    unittest.main()
