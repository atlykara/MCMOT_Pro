"""Faz 3 icin yon bazli, zaman kontrollu FIFO handoff eslestirici."""

from collections import deque
from dataclasses import dataclass


DIRECTIONS = ("camA_to_camB", "camB_to_camA")


@dataclass(frozen=True)
class TrackEvent:
    timestamp: float
    camera_id: str
    track_id: int
    vehicle_class: str
    movement_label: str
    event_type: str


@dataclass(frozen=True)
class Match:
    match_id: str
    movement_label: str
    source: TrackEvent
    target: TrackEvent
    delta_t: float


class FifoMatcher:
    """Her fiziksel hareket yonu icin ayri bir cikis kuyrugu tutar.

    Kamera ureticileri exit/entry olaylarini bu sinifa yollar. Kaynak exit olayi
    kuyruga eklenir; hedef entry olayi yalnizca kuyrugun basindaki uygun kaynakla
    eslesir. Zaman asimina ugrayan kaynak once dusurulur, boylece tek bir kacirma
    sonraki tum araclari kalici olarak bir sira kaydirmaz.
    """

    def __init__(self, delay_s=0.7, window_width_s=0.8, max_queue_size=30,
                 history_size=10):
        if delay_s < 0 or window_width_s <= 0:
            raise ValueError("delay_s >= 0 ve window_width_s > 0 olmali")
        self.delay_s = float(delay_s)
        self.max_delay_s = float(delay_s + window_width_s)
        self.max_queue_size = int(max_queue_size)
        self.queues = {direction: deque() for direction in DIRECTIONS}
        self.matches = []
        self.expired = []
        self.unmatched_entries = []
        self.recent = deque(maxlen=int(history_size))
        self._next_match_id = 1

    def process(self, event):
        if event.movement_label not in self.queues:
            raise ValueError(f"Bilinmeyen hareket yonu: {event.movement_label}")
        self.expire(event.timestamp)
        if event.event_type == "exit":
            return self._enqueue(event)
        if event.event_type == "entry":
            return self._consume(event)
        raise ValueError(f"Bilinmeyen event_type: {event.event_type}")

    def expire(self, now):
        for direction, queue in self.queues.items():
            while queue and now - queue[0].timestamp > self.max_delay_s:
                event = queue.popleft()
                self.expired.append(event)
                self.recent.appendleft(
                    f"TIMEOUT {direction}: {event.camera_id}#{event.track_id}"
                )

    def _enqueue(self, event):
        queue = self.queues[event.movement_label]
        if len(queue) >= self.max_queue_size:
            dropped = queue.popleft()
            self.expired.append(dropped)
            self.recent.appendleft(
                f"OVERFLOW {event.movement_label}: {dropped.camera_id}#{dropped.track_id}"
            )
        queue.append(event)
        self.recent.appendleft(
            f"FIFO + {event.movement_label}: {event.camera_id}#{event.track_id}"
        )
        return None

    def _consume(self, target):
        queue = self.queues[target.movement_label]
        if not queue:
            self.unmatched_entries.append(target)
            self.recent.appendleft(
                f"NO SOURCE {target.movement_label}: {target.camera_id}#{target.track_id}"
            )
            return None

        source = queue[0]
        delta_t = target.timestamp - source.timestamp
        if delta_t < self.delay_s:
            self.unmatched_entries.append(target)
            self.recent.appendleft(
                f"EARLY {target.movement_label}: {target.camera_id}#{target.track_id}"
            )
            return None

        source = queue.popleft()
        match = Match(
            match_id=f"M{self._next_match_id:06d}",
            movement_label=target.movement_label,
            source=source,
            target=target,
            delta_t=round(delta_t, 3),
        )
        self._next_match_id += 1
        self.matches.append(match)
        self.recent.appendleft(
            f"MATCH {match.match_id}: {source.camera_id}#{source.track_id} -> "
            f"{target.camera_id}#{target.track_id} ({match.delta_t:.2f}s)"
        )
        return match

    def queue_snapshot(self, direction, now):
        return [
            {
                "camera_id": event.camera_id,
                "track_id": event.track_id,
                "vehicle_class": event.vehicle_class,
                "age_s": round(now - event.timestamp, 2),
            }
            for event in self.queues[direction]
        ]

