# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent
import gc
import logging
import os
import signal
import threading
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from queue import Queue
from typing import Callable, Dict

from .exception import FedLocalError, FedRemoteError, main_thread_assert
from .object import FedFuture, FedObject
from .proxy import SenderReceiverProxy

logger = logging.getLogger(__name__)


class GlobalContext:
    def __init__(
        self,
        job_name: str,
        current_party: str,
        proxy: SenderReceiverProxy,
        addresses: Dict,
    ) -> None:
        self._task_executor: Executor = ThreadPoolExecutor()
        proxy_workers_limit = None if proxy.concurrent() else 1
        self._send_executor: Executor = ThreadPoolExecutor(
            max_workers=proxy_workers_limit
        )
        self._recv_executor: Executor = ThreadPoolExecutor(
            max_workers=proxy_workers_limit
        )
        self._proxy: SenderReceiverProxy = proxy
        self._job_name = job_name
        # fedobj seq id
        self._seq_count = 0
        # debug logging only
        self._task_count = 0
        self._current_party = current_party
        self._addresses = addresses

        self._lock = threading.Lock()
        self._remote_exception = None
        self._local_exception = None
        self._send_exception = None
        self._atomic_shutdown_flag = True
        self._clean_queue = Queue()
        self._clean_stopped = False
        self._clean_thread = threading.Thread(target=self._cleanup)
        self._clean_thread.start()

    def get_addresses(self):
        return self._addresses

    def set_remote_exception(self, exce):
        with self._lock:
            self._remote_exception = exce

    def get_remote_exception(self):
        with self._lock:
            return self._remote_exception

    def set_local_exception(self, exec):
        with self._lock:
            self._local_exception = exec

    def get_local_exception(self):
        with self._lock:
            return self._local_exception

    def set_send_exception(self, exec):
        with self._lock:
            self._send_exception = exec

    def get_send_exception(self):
        with self._lock:
            return self._send_exception

    def get_last_exception(self):
        with self._lock:
            if self._remote_exception is not None:
                return self._remote_exception
            if self._local_exception is not None:
                return self._local_exception
            if self._send_exception is not None:
                return self._send_exception
        return None

    def get_job_name(self) -> str:
        return self._job_name

    def get_party(self) -> str:
        return self._current_party

    def next_seq_id(self) -> int:
        self._seq_count += 1
        return self._seq_count

    def next_task_id(self) -> int:
        self._task_count += 1
        return self._task_count

    def submit_task(self, fn: Callable, *args, **kwargs) -> Future:
        def _on_task_finish(future: Future):
            try:
                exception = future.exception()
                if exception:
                    self.set_local_exception(exception)
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                self.set_local_exception(e)

        future = self._task_executor.submit(fn, *args, **kwargs)
        future.add_done_callback(_on_task_finish)
        return future

    def acquire_shutdown_flag(self) -> bool:
        with self._lock:
            if self._atomic_shutdown_flag:
                self._atomic_shutdown_flag = False
                return True
            return False

    def _signal_exit(self):
        """
        Exit the current process immediately. The signal will be captured
        in main thread where the `stop` will be called.
        """
        if self.acquire_shutdown_flag():
            logger.warning("Signal SIGINT to exit.")
            os.kill(os.getpid(), signal.SIGINT)

    def _cleanup(self):
        while True:
            task = self._clean_queue.get()
            if not self._clean_stopped and callable(task):
                if not task():
                    break
            else:
                break

    def send(self, target_party: str, fed_obj: FedObject) -> None:
        main_thread_assert()
        if fed_obj.is_sent(target_party):
            return
        else:
            fed_obj.mark_send(target_party)

        seq_id = fed_obj.get_seq_id()

        def _send():
            local_err = None
            try:
                logger.debug(f"Send try get_data from {fed_obj}")
                obj = fed_obj.get_object()
                logger.debug(f"Send done get_data from {fed_obj}")
            except FedRemoteError:
                raise
            except Exception as e:
                local_err = e if isinstance(e, FedLocalError) else FedLocalError(e)
                obj = FedRemoteError(self.get_party(), e)
            logger.debug(f"try send obj for {fed_obj} to {target_party}")
            self._proxy.send(target_party, obj, seq_id)
            logger.debug(f"done send obj for {fed_obj} to {target_party}")
            if local_err:
                raise local_err

        logger.debug(f"try submit send task for {fed_obj} to {target_party}")
        send_future = self._send_executor.submit(_send)

        def _send_check():
            try:
                send_future.result()
                return True
            except FedRemoteError as e:
                logger.info(f"FedRemoteError on seq id {seq_id}")
            except FedLocalError as e:
                logger.info(f"FedLocalError on seq id {seq_id}")
            except Exception as e:
                logger.exception(f"Failed to send seq id {seq_id} to {target_party}")
                self.set_send_exception(e)
            self._signal_exit()
            return False

        self._clean_queue.put(_send_check)

    def recv(self, fed_obj: FedObject) -> FedObject:
        main_thread_assert()
        if fed_obj.has_object():
            return fed_obj

        src_party = fed_obj.get_party()
        seq_id = fed_obj.get_seq_id()

        def _recv():
            logger.debug(f"Try recv from {src_party} with seq id {seq_id}")
            data = self._proxy.recv(src_party, seq_id)
            if isinstance(data, FedRemoteError):
                logger.error(
                    f"Receiving exception: {type(data)}, {data} from {src_party}, "
                    f"seq id {seq_id}. Re-raise it."
                )
                self.set_remote_exception(data)
                raise data
            logger.debug(f"Done recv from {src_party} with seq id {seq_id}")
            return data

        logger.debug(f"try submit recv task for {fed_obj}")
        future = self._recv_executor.submit(_recv)
        fed_obj.set_object(FedFuture(future))
        return fed_obj

    def stop(self, wait_for_sending=True, on_error=False):
        main_thread_assert()
        logger.info(
            f"Try stop context, wait_for_sending {wait_for_sending}, on_error {on_error}"
        )
        self._task_executor.shutdown(wait=not on_error, cancel_futures=on_error)
        logger.info("task_executor stopped")
        self._recv_executor.shutdown(wait=not on_error, cancel_futures=on_error)
        logger.info("recv_executor stopped")
        self._send_executor.shutdown(
            wait=wait_for_sending, cancel_futures=not wait_for_sending
        )
        logger.info("send_executor stopped")

        if not wait_for_sending or on_error:
            self._clean_stopped = True
        self._clean_queue.put("STOP")
        if not on_error:
            self._clean_thread.join()
            logger.info("clean_thread stopped")

        if not on_error:
            self._proxy.stop()
            logger.info("proxy stopped")

        logger.info("Context stopped")


_global_context: GlobalContext = None


def init_global_context(
    job_name: str,
    current_party: str,
    proxy: SenderReceiverProxy,
    addresses: Dict,
) -> None:
    main_thread_assert()
    global _global_context
    if _global_context is None:
        _global_context = GlobalContext(
            job_name,
            current_party,
            proxy,
            addresses,
        )


def get_global_context(assert_none=True) -> GlobalContext:
    global _global_context
    if assert_none and _global_context is None:
        raise RuntimeError("sf_fed is stoped or not init")
    return _global_context


def clear_global_context(wait_for_sending=False, on_error=False):
    main_thread_assert()
    global _global_context
    if _global_context is not None:
        _global_context.stop(wait_for_sending, on_error)
        _global_context = None
        gc.collect()
