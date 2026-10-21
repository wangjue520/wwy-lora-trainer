"""Optional TensorBoard writing in a bounded, disposable CPU process."""
from __future__ import annotations

import atexit
import multiprocessing
from pathlib import Path
import queue
import sys


def _warn(message):
    try:
        print(f"TensorBoard: {message}; training continues.", file=sys.stderr, flush=True)
    except Exception:
        pass


def _writer_process(messages, directory, purge_step):
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(directory, purge_step=purge_step, flush_secs=10)
        while True:
            message = messages.get()
            if message is None:
                break
            kind, step, values = message
            if kind == "scalars":
                for tag, value in values.items():
                    writer.add_scalar(tag, value, step)
            elif kind == "image":
                import numpy as np
                from PIL import Image
                with Image.open(values) as image:
                    writer.add_image("sample/preview", np.array(image.convert("RGB")), step, dataformats="HWC")
    except Exception as error:
        _warn(f"writer stopped ({type(error).__name__}: {error})")
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass


class TensorBoardLogger:
    """Never wait for logging in the training loop, and never propagate errors."""

    def __init__(self, directory, *, enabled=True, start_step=0):
        self.process = None
        self.messages = None
        self.active = False
        self.warned_full = False
        if not enabled:
            return
        try:
            context = multiprocessing.get_context("spawn")
            self.messages = context.Queue(maxsize=128)
            self.process = context.Process(
                target=_writer_process,
                args=(self.messages, str(Path(directory)), start_step + 1 if start_step else None),
                name="moss-tensorboard", daemon=True,
            )
            self.process.start()
            self.active = True
            atexit.register(self.close)
        except Exception as error:
            _warn(f"disabled ({type(error).__name__}: {error})")
            self.close()

    def _send(self, message):
        if not self.active:
            return
        try:
            if not self.process.is_alive():
                self.active = False
                _warn("writer process exited; logging disabled")
                return
            self.messages.put_nowait(message)
        except queue.Full:
            if not self.warned_full:
                _warn("queue full; dropping log entries instead of blocking")
                self.warned_full = True
        except Exception as error:
            self.active = False
            _warn(f"logging disabled ({type(error).__name__}: {error})")

    def scalars(self, step, values):
        try:
            self._send(("scalars", int(step), {str(k): float(v) for k, v in values.items()}))
        except Exception as error:
            self.active = False
            _warn(f"invalid log entry ({error})")

    def image(self, step, path):
        try:
            self._send(("image", int(step), str(path)))
        except Exception as error:
            self.active = False
            _warn(f"image logging disabled ({error})")

    def close(self):
        self.active = False
        try:
            atexit.unregister(self.close)
            if self.process is not None:
                if self.process.is_alive():
                    try:
                        self.messages.put_nowait(None)
                    except Exception:
                        pass
                    self.process.join(timeout=1.0)
                    if self.process.is_alive():
                        self.process.terminate()
                        self.process.join(timeout=0.5)
                self.process.close()
                self.process = None
            if self.messages is not None:
                self.messages.cancel_join_thread()
                self.messages.close()
                self.messages = None
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
