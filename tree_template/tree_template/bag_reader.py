"""
bag_reader.py

Reads messages from a rosbag2 bag (sqlite3 or mcap) in a receive-time window, with rosbag2_py.
"""

import glob
import os
from typing import Iterable, Iterator, Optional, Tuple

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


class BagReader:
    def __init__(self, uri: str):
        storage = "mcap" if uri.endswith(".mcap") or glob.glob(os.path.join(uri, "*.mcap")) else "sqlite3"
        self._reader = rosbag2_py.SequentialReader()
        self._reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id=storage),
                          rosbag2_py.ConverterOptions("cdr", "cdr"))
        self.types = {t.name: t.type for t in self._reader.get_all_topics_and_types()}
        md = self._reader.get_metadata()
        self.start_ns = md.starting_time.nanoseconds
        self.end_ns = self.start_ns + md.duration.nanoseconds

    def raw(self, topics: Iterable[str], start_ns: Optional[int] = None,
            end_ns: Optional[int] = None) -> Iterator[Tuple[str, bytes, int]]:
        """(topic, serialized data, receive time ns) of the messages on topics in [start_ns, end_ns]."""
        topics = [t for t in topics if t in self.types]
        if not topics:
            return
        self._reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
        self._reader.seek(self.start_ns if start_ns is None else max(int(start_ns), 0))
        while self._reader.has_next():
            topic, data, t = self._reader.read_next()
            if end_ns is not None and t > end_ns:
                return
            yield topic, data, t

    def deserialize(self, topic: str, data: bytes):
        return deserialize_message(data, get_message(self.types[topic]))

    def messages(self, topics: Iterable[str], start_ns: Optional[int] = None,
                 end_ns: Optional[int] = None) -> Iterator[Tuple[str, object, int]]:
        """(topic, message, receive time ns) of the messages on topics in [start_ns, end_ns]."""
        for topic, data, t in self.raw(topics, start_ns, end_ns):
            yield topic, self.deserialize(topic, data), t

    def first(self, topic: str, start_ns: Optional[int] = None):
        """(message, receive time ns) of the first message on topic from start_ns, or (None, None)."""
        for _, msg, t in self.messages([topic], start_ns):
            return msg, t
        return None, None

    def last(self, topic: str, window_ns: int = int(10e9)):
        """(message, receive time ns) of the last message on topic, or (None, None).
        Reads the last window_ns of the bag first, and all the bag if the topic has no message there."""
        for start in (self.end_ns - window_ns, None):
            last = None
            for _, data, t in self.raw([topic], start):
                last = (data, t)
            if last:
                return self.deserialize(topic, last[0]), last[1]
        return None, None
