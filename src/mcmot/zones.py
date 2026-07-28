"""Faz 2 zone/ROI yardimcilari.

Bu modulun ana sorusu:
    "Bir aracin foot_point noktasi cizdigimiz poligonun icinde mi?"

Dis bagimlilik kullanmadan basit ray-casting point-in-polygon uygular.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


Point = tuple[float, float]


@dataclass(frozen=True)
class Zone:
    zone_id: str
    polygon: list[Point]
    kind: str = "roi"
    notes: str = ""


@dataclass(frozen=True)
class ZoneConfig:
    camera_id: str
    image_size: tuple[int, int]
    zones: list[Zone]


def point_in_polygon(point: Point, polygon: Iterable[Point]) -> bool:
    """Ray-casting algoritmasi ile nokta poligon icinde mi bakar.

    Koordinat sistemi OpenCV/goruntu koordinatidir: x saga, y asagi artar.
    Kenara tam denk gelen noktalar pratikte iceride kabul edilir.
    """
    x, y = point
    pts = list(polygon)
    if len(pts) < 3:
        return False

    inside = False
    j = len(pts) - 1
    for i, (xi, yi) in enumerate(pts):
        xj, yj = pts[j]

        # Nokta kenar ustunde mi? Kucuk toleransla iceride say.
        cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
        if abs(cross) < 1e-9:
            if min(xi, xj) <= x <= max(xi, xj) and min(yi, yj) <= y <= max(yi, yj):
                return True

        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i

    return inside


def load_zone_config(path: Path) -> ZoneConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    zones = []
    for item in raw.get("zones", []):
        zones.append(
            Zone(
                zone_id=str(item["zone_id"]),
                polygon=[(float(x), float(y)) for x, y in item["polygon"]],
                kind=str(item.get("kind", "roi")),
                notes=str(item.get("notes", "")),
            )
        )

    image_size = raw.get("image_size") or [0, 0]
    return ZoneConfig(
        camera_id=str(raw["camera_id"]),
        image_size=(int(image_size[0]), int(image_size[1])),
        zones=zones,
    )


def zones_for_point(point: Point, zones: Iterable[Zone]) -> list[Zone]:
    return [zone for zone in zones if point_in_polygon(point, zone.polygon)]


def make_zone_yaml(camera_id: str, image_size: tuple[int, int], zones: list[dict]) -> dict:
    """02_define_zones.py tarafindan yazilacak YAML govdesini olusturur."""
    return {
        "camera_id": camera_id,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "zones": zones,
    }
