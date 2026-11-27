"""
Tests for the Audacity label file import endpoint.
"""
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.models.annotation import Annotation
from tests.api.routes.test_annotations import create_test_media


def _audacity_import(
    client: TestClient,
    superuser_token_headers: dict[str, str],
    project_id: int,
    collection_id: int,
    files: list[tuple[str, str]],
    dry_run: bool = True,
):
    return client.post(
        f"{settings.API_V1_STR}/annotations/audacity-imports",
        headers=superuser_token_headers,
        data={
            "project_id": str(project_id),
            "collection_id": str(collection_id),
            "dry_run": "true" if dry_run else "false",
        },
        files=[
            ("files", (name, content.encode(), "text/plain"))
            for name, content in files
        ],
    )


def test_import_audacity_region_and_point_labels(
    client: TestClient,
    superuser_token_headers: dict[str, str],
    db: Session,
) -> None:
    media, col, project = create_test_media(db, media_type="audio", is_metadata=False)
    label_name = f"{Path(media.name).stem}.txt"
    content = (
        "0.5\t2.3\tBird call\n"
        "3.1\n"
        "\n"
    )

    preview = _audacity_import(
        client, superuser_token_headers, project.project_id, col.collection_id,
        [(label_name, content)],
    )
    assert preview.status_code == 200
    report = preview.json()["data"]
    assert report["failed"] == 0
    assert report["succeeded"] == 2
    assert report["dry_run"] is True

    committed = _audacity_import(
        client, superuser_token_headers, project.project_id, col.collection_id,
        [(label_name, content)],
        dry_run=False,
    )
    assert committed.status_code == 200
    assert committed.json()["data"]["committed"] is True

    annotations = db.exec(
        select(Annotation)
        .where(Annotation.media_id == media.media_id)
        .order_by(Annotation.annotation_id)
    ).all()
    assert len(annotations) == 2
    region, point = annotations
    assert region.min_x == 0.5
    assert region.max_x == 2.3
    assert region.comments == "Bird call"
    assert region.min_y == 0.0
    assert region.max_y == 8000.0
    assert region.sound_id is None
    assert region.object_type is None
    assert region.individual_num == 1
    assert point.min_x == 3.1
    assert point.max_x == 3.1
    assert point.comments is None


def test_import_audacity_spectral_labels_use_frequency_bounds(
    client: TestClient,
    superuser_token_headers: dict[str, str],
    db: Session,
) -> None:
    media, col, project = create_test_media(db, media_type="audio", is_metadata=False)
    label_name = f"{Path(media.name).stem}.txt"
    content = "1.0\t2.0\t1200\t3400\tCrustacean snaps\n"

    response = _audacity_import(
        client, superuser_token_headers, project.project_id, col.collection_id,
        [(label_name, content)],
        dry_run=False,
    )
    assert response.status_code == 200
    assert response.json()["data"]["committed"] is True

    annotation = db.exec(
        select(Annotation).where(Annotation.media_id == media.media_id)
    ).one()
    assert annotation.min_y == 1200.0
    assert annotation.max_y == 3400.0
    assert annotation.comments == "Crustacean snaps"


def test_import_audacity_supports_multiple_files_and_media_matching(
    client: TestClient,
    superuser_token_headers: dict[str, str],
    db: Session,
) -> None:
    media_a, col, project = create_test_media(db, media_type="audio", is_metadata=False)
    media_b, _col2, _project2 = create_test_media(db, media_type="audio", is_metadata=False)
    from app.models.media import MediaCollection

    db.add(MediaCollection(media_id=media_b.media_id, collection_id=col.collection_id, added_by=1))
    db.commit()

    label_file_a = f"{Path(media_a.name).stem}.txt"
    label_file_b = f"{Path(media_b.name).stem}.txt"
    response = _audacity_import(
        client, superuser_token_headers, project.project_id, col.collection_id,
        [
            (label_file_a, "0.0\t1.0\tLabel A\n"),
            (label_file_b, "2.0\tLabel B\n"),
        ],
        dry_run=False,
    )
    assert response.status_code == 200
    assert response.json()["data"]["succeeded"] == 2

    comments_a = db.exec(
        select(Annotation.comments).where(Annotation.media_id == media_a.media_id)
    ).all()
    comments_b = db.exec(
        select(Annotation.comments).where(Annotation.media_id == media_b.media_id)
    ).all()
    assert comments_a == ["Label A"]
    assert comments_b == ["Label B"]


def test_import_audacity_rejects_unknown_file_names_atomically(
    client: TestClient,
    superuser_token_headers: dict[str, str],
    db: Session,
) -> None:
    media, col, project = create_test_media(db, media_type="audio", is_metadata=False)
    files = [
        (f"{Path(media.name).stem}.txt", "0.5\t1.0\tKnown media\n"),
        ("unknown_recording.txt", "0.5\t1.0\tOrphan label\n"),
    ]

    committed = _audacity_import(
        client, superuser_token_headers, project.project_id, col.collection_id,
        files,
        dry_run=False,
    )
    assert committed.status_code == 200
    report = committed.json()["data"]
    assert report["failed"] == 1
    assert report["committed"] is False
    assert "unknown_recording" in report["rows"][1]["reason"]
    assert db.exec(
        select(Annotation).where(Annotation.media_id == media.media_id)
    ).first() is None


def test_import_audacity_reports_invalid_rows(
    client: TestClient,
    superuser_token_headers: dict[str, str],
    db: Session,
) -> None:
    media, col, project = create_test_media(db, media_type="audio", is_metadata=False)
    label_name = f"{Path(media.name).stem}.txt"
    content = (
        "2.0\t1.0\tEnd before start\n"
        "not-a-time\tLabel\n"
        "1.0\t2.0\t1000\tWrong column count\n"
    )

    response = _audacity_import(
        client, superuser_token_headers, project.project_id, col.collection_id,
        [(label_name, content)],
    )
    assert response.status_code == 200
    report = response.json()["data"]
    assert report["failed"] == 3
    reasons = " | ".join(row["reason"] for row in report["rows"])
    assert "end time precedes" in reasons
    assert "not a valid time value" in reasons
    assert "expected 1, 2, 3, or 5 columns" in reasons


def test_import_audacity_requires_write_permission(
    client: TestClient,
    db: Session,
    normal_user_token_headers: dict[str, str],
) -> None:
    media, col, project = create_test_media(db, media_type="audio", is_metadata=False)
    response = _audacity_import(
        client, normal_user_token_headers, project.project_id, col.collection_id,
        [(f"{Path(media.name).stem}.txt", "0.5\t1.0\tLabel\n")],
    )
    assert response.status_code == 403
