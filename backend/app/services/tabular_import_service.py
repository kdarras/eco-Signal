from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Hashable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlmodel import Field, Session, SQLModel, select

from app.csv_import import (
    ImportResult,
    ImportRowResult,
    _decode_import_content,
    effective_header_width,
    ensure_row_width,
    parse_csv,
    read_cell,
    resolve_header_positions,
)
from app.models import Annotation, Media, MediaCollection, Project, ProjectCollection, User
from app.schemas.annotation import AnnotationCreate
from app.schemas.collection import CollectionCreate
from app.schemas.index_log import IndexLogCreateRequest
from app.schemas.project import ProjectCreate
from app.schemas.review import ReviewCreate
from app.schemas.site import SiteCreate
from app.schemas.user import UserCreate
from app.services import (
    analysis_service,
    annotation_service,
    collection_service,
    project_service,
    review_service,
    site_service,
    task_service,
    user_service,
)


class TaskImportRow(SQLModel):
    media_id: int = Field(gt=0)
    type: Literal["media", "annotation"]
    annotation_id: int | None = Field(default=None, gt=0)
    assignee_id: int = Field(gt=0)
    comment: str | None = Field(default=None, max_length=1000)


def _require_media_scope(
    session: Session,
    media_id: int,
    project_id: int,
    collection_id: int,
) -> None:
    scoped_media_id = session.exec(
        select(MediaCollection.media_id)
        .join(
            ProjectCollection,
            ProjectCollection.collection_id == MediaCollection.collection_id,
        )
        .where(
            MediaCollection.media_id == media_id,
            MediaCollection.collection_id == collection_id,
            ProjectCollection.project_id == project_id,
        )
    ).first()
    if scoped_media_id is None:
        raise HTTPException(
            status_code=422,
            detail="Media does not belong to the selected project and collection",
        )


def _annotation_media_id(session: Session, annotation_id: int) -> int:
    media_id = session.exec(
        select(Annotation.media_id).where(Annotation.annotation_id == annotation_id)
    ).first()
    if media_id is None:
        raise HTTPException(status_code=422, detail="Annotation not found")
    return media_id


def _require_annotation_media_type(
    session: Session,
    media_id: int,
    expected_media_type: Literal["audio", "photo"] | None,
) -> None:
    if expected_media_type is None:
        return
    media = session.get(Media, media_id)
    if media is None or media.media_type != expected_media_type:
        actual = media.media_type if media is not None else "unknown"
        raise HTTPException(
            status_code=422,
            detail=(
                f"Media {media_id} is {actual}, but this import requires "
                f"{expected_media_type} media"
            ),
        )


def _parse_json_cell(value: str | None, field: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} must contain a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return parsed


def _parse_models(
    text: str,
    schema: type[SQLModel],
    *,
    injected: dict[str, Any] | None = None,
    converters: dict[str, Callable[[str | None], Any]] | None = None,
) -> tuple[ImportResult, list[tuple[int, SQLModel]]]:
    injected = injected or {}
    converters = converters or {}
    report = ImportResult()
    try:
        parsed = parse_csv(text)
    except HTTPException as exc:
        report.global_errors.append(str(exc.detail))
        return report.finalize(), []
    if not parsed:
        report.global_errors.append("Import file contains no records")
        return report.finalize(), []

    fields = {
        name: field
        for name, field in schema.model_fields.items()
        if name not in injected
    }
    field_headers = {name: name for name in fields}
    required = [
        name
        for name, field in fields.items()
        if field.is_required()
    ]
    header, *data_rows = parsed
    try:
        width = effective_header_width(header)
        positions = resolve_header_positions(header, field_headers, required)
    except HTTPException as exc:
        report.global_errors.append(str(exc.detail))
        report.reject_data_rows(data_rows, str(exc.detail))
        return report.finalize(), []

    valid: list[tuple[int, SQLModel]] = []
    for row_number, row in enumerate(data_rows, start=2):
        if not row or not any(value.strip() for value in row):
            report.rows.append(ImportRowResult(row_number=row_number, status="skipped", reason="Blank row"))
            continue
        try:
            ensure_row_width(row, row_number, width)
            payload: dict[str, Any] = {
                name: (read_cell(row, positions, name) or None)
                for name in fields
                if name in positions
            }
            for name, converter in converters.items():
                if name in payload:
                    payload[name] = converter(payload[name])
            payload.update(injected)
            item = schema.model_validate(payload)
        except HTTPException as exc:
            report.rows.append(ImportRowResult(row_number=row_number, status="failed", reason=str(exc.detail)))
        except (ValidationError, ValueError) as exc:
            if isinstance(exc, ValidationError):
                error = exc.errors()[0]
                field = str(error["loc"][-1])
                reason = str(error["msg"])
            else:
                field = None
                reason = str(exc)
            report.rows.append(ImportRowResult(row_number=row_number, status="failed", field=field, reason=reason))
        else:
            valid.append((row_number, item))
            report.rows.append(ImportRowResult(row_number=row_number, status="succeeded"))
    return report.finalize(), valid


def _execute(
    session: Session,
    report: ImportResult,
    rows: list[tuple[int, SQLModel]],
    writer: Callable[[SQLModel], None],
    *,
    dry_run: bool,
    key: Callable[[SQLModel], Hashable] | None = None,
    preflight: Callable[[SQLModel], None] | None = None,
) -> ImportResult:
    results = {row.row_number: row for row in report.rows}
    if report.global_errors or report.failed:
        if not dry_run:
            report.reject_candidates()
        return report

    accepted: list[tuple[int, SQLModel]] = []
    seen: set[Hashable] = set()
    for row_number, item in rows:
        if key is not None:
            item_key = key(item)
            if item_key in seen:
                result = results[row_number]
                result.status = "skipped"
                result.reason = "Duplicate record in file"
                continue
            seen.add(item_key)

        try:
            if preflight is not None:
                preflight(item)
        except HTTPException as exc:
            result = results[row_number]
            if exc.status_code == 409 or "already exists" in str(exc.detail).lower():
                result.status = "skipped"
            else:
                result.status = "failed"
            result.reason = str(exc.detail)
        except ValueError as exc:
            result = results[row_number]
            result.status = "failed"
            result.reason = str(exc)
        else:
            accepted.append((row_number, item))

    report.finalize()
    if report.failed:
        if not dry_run:
            report.reject_candidates()
        return report
    if dry_run:
        return report.finalize()

    def fail_write(row_number: int, reason: str) -> ImportResult:
        results[row_number].status = "failed"
        results[row_number].reason = reason
        session.rollback()
        report.reject_candidates()
        return report.finalize()

    try:
        for row_number, item in accepted:
            if results[row_number].status != "succeeded":
                continue
            try:
                writer(item)
            except HTTPException as exc:
                return fail_write(row_number, str(exc.detail))
            except IntegrityError:
                return fail_write(row_number, "Record violates a data constraint")
            except ValueError as exc:
                return fail_write(row_number, str(exc))
        session.commit()
    except Exception:
        session.rollback()
        report.global_errors.append("No data was written because the import transaction failed")
        report.reject_candidates()
        return report
    report.committed = True
    return report.finalize()


def import_projects(session: Session, text: str, user: User, *, dry_run: bool) -> ImportResult:
    report, rows = _parse_models(text, ProjectCreate)
    existing = {
        project.name.strip().casefold(): project
        for project in session.exec(select(Project)).all()
        if project.name.strip()
    }
    results = {row.row_number: row for row in report.rows}
    pending: list[tuple[int, SQLModel]] = []
    for row_number, item in rows:
        key = str(item.name).strip().casefold()
        project = existing.get(key)
        if project is None:
            pending.append((row_number, item))
            continue
        values = item.model_dump()
        exact_duplicate = all(
            (
                str(getattr(project, field, "")).strip().casefold()
                if field == "name"
                else getattr(project, field, None)
            )
            == (key if field == "name" else value)
            for field, value in values.items()
        )
        result = results[row_number]
        result.field = "name"
        if exact_duplicate:
            result.status = "skipped"
            result.reason = "Project already exists"
        else:
            result.status = "failed"
            result.reason = "Project name conflicts with an existing record"
    report.finalize()
    return _execute(
        session,
        report,
        pending,
        lambda item: project_service.create_project(session, item, user, commit=False),
        dry_run=dry_run,
        key=lambda item: str(item.name).strip().casefold(),
        preflight=lambda item: project_service.validate_project_create(session, item),
    )


def import_collections(
    session: Session,
    text: str,
    user: User,
    project_id: int,
    *,
    dry_run: bool,
) -> ImportResult:
    report, rows = _parse_models(text, CollectionCreate)
    return _execute(
        session,
        report,
        rows,
        lambda item: collection_service.create_collection(
            session, item, user, project_id, commit=False
        ),
        dry_run=dry_run,
        key=lambda item: str(item.name).strip().casefold(),
        preflight=lambda item: collection_service.validate_collection_create(session, item, project_id),
    )


def import_sites(
    session: Session,
    text: str,
    user: User,
    project_id: int,
    collection_id: int,
    *,
    dry_run: bool,
) -> ImportResult:
    report, rows = _parse_models(
        text,
        SiteCreate,
        injected={"project_id": project_id, "collection_id": collection_id},
    )
    return _execute(
        session,
        report,
        rows,
        lambda item: site_service.create_site(session, item, user, commit=False),
        dry_run=dry_run,
        key=lambda item: str(item.name).strip().casefold(),
        preflight=lambda item: site_service.validate_site_create(session, item, user),
    )


def import_annotations(
    session: Session,
    text: str,
    user: User,
    project_id: int,
    collection_id: int,
    *,
    dry_run: bool,
    expected_media_type: Literal["audio", "photo"] | None = None,
) -> ImportResult:
    report, rows = _parse_models(text, AnnotationCreate, injected={"project_id": project_id})
    return _execute(
        session,
        report,
        rows,
        lambda item: (
            _require_media_scope(
                session, item.media_id, project_id, collection_id
            ),
            _require_annotation_media_type(session, item.media_id, expected_media_type),
            annotation_service.create_annotation(session, user, item, commit=False),
        ),
        dry_run=dry_run,
        key=lambda item: (
            item.media_id,
            item.min_x,
            item.max_x,
            item.min_y,
            item.max_y,
            item.taxon_id,
            item.sound_id,
        ),
        preflight=lambda item: (
            _require_media_scope(session, item.media_id, project_id, collection_id),
            _require_annotation_media_type(session, item.media_id, expected_media_type),
            annotation_service.validate_annotation_create(session, user, item),
        ),
    )


_AUDACITY_IMPORT_EXTENSIONS = {"txt", "csv"}
_MEDIA_FILE_EXTENSIONS = {
    ".wav", ".wave", ".aif", ".aiff", ".aifc", ".flac", ".mp3",
    ".ogg", ".oga", ".opus", ".m4a", ".wma", ".aac",
}
_MAX_ANNOTATION_COMMENTS_LENGTH = 500
_DEFAULT_NYQUIST_HZ = 24000.0


class _AudacityLabel:
    """One parsed Audacity label row awaiting media resolution."""

    __slots__ = ("filename", "row_number", "media_keys", "start", "end", "low", "high", "label")

    def __init__(
        self,
        filename: str,
        row_number: int,
        media_keys: list[str],
        start: float,
        end: float | None,
        low: float | None,
        high: float | None,
        label: str | None,
    ) -> None:
        self.filename = filename
        self.row_number = row_number
        self.media_keys = media_keys
        self.start = start
        self.end = end
        self.low = low
        self.high = high
        self.label = label


def _audacity_name_keys(name: str | None) -> set[str]:
    """Case-folded file-name keys used to link label files to media."""
    if not name or not name.strip():
        return set()
    stem = Path(name.strip()).stem.strip().casefold()
    keys = {stem} if stem else set()
    for extension in _MEDIA_FILE_EXTENSIONS:
        if stem.endswith(extension):
            stripped = stem[: -len(extension)].strip()
            if stripped:
                keys.add(stripped)
            break
    return keys


def _audacity_timestamp(raw: str, filename: str, row_number: int) -> float:
    """Parse a numeric seconds value or an HH:MM:SS(.fraction) timestamp."""
    text = raw.strip()
    try:
        if ":" in text:
            parts = text.split(":")
            if len(parts) > 3:
                raise ValueError
            seconds = 0.0
            for part in parts:
                seconds = seconds * 60.0 + float(part)
        else:
            seconds = float(text)
    except ValueError as exc:
        raise ValueError(f"{filename}: row {row_number}: '{text}' is not a valid time value") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{filename}: row {row_number}: time values must be non-negative numbers")
    return seconds


def _audacity_frequency(raw: str | None, filename: str, row_number: int, column: str) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{filename}: row {row_number}: '{text}' is not a valid {column} value") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{filename}: row {row_number}: {column} must be a non-negative number")
    return value


def _parse_audacity_line(line: str, filename: str, row_number: int) -> _AudacityLabel:
    """Parse one Audacity label row: start[tab]end[tab]label, start[tab]label, or the spectral 5-column form."""
    cells = next(csv.reader([line], delimiter="\t"))
    if len(cells) < 2:
        cells = next(csv.reader([line]))
    cells = [cell.strip() for cell in cells]
    if len(cells) == 1:
        start_raw, end_raw, low_raw, high_raw, label = cells[0], None, None, None, None
    elif len(cells) == 2:
        start_raw, end_raw, low_raw, high_raw, label = cells[0], None, None, None, cells[1]
    elif len(cells) == 3:
        start_raw, end_raw, label = cells
        low_raw = high_raw = None
    elif len(cells) == 5:
        start_raw, end_raw, low_raw, high_raw, label = cells
    else:
        raise ValueError(
            f"{filename}: row {row_number}: expected 1, 2, 3, or 5 columns "
            "(start, [end], [low frequency, high frequency], label)"
        )
    start = _audacity_timestamp(start_raw, filename, row_number)
    end = _audacity_timestamp(end_raw, filename, row_number) if end_raw else None
    low = _audacity_frequency(low_raw, filename, row_number, "low frequency")
    high = _audacity_frequency(high_raw, filename, row_number, "high frequency")
    if end is not None and end < start:
        raise ValueError(f"{filename}: row {row_number}: the label end time precedes its start time")
    if (low is None) != (high is None):
        raise ValueError(f"{filename}: row {row_number}: frequency bounds require both low and high values")
    if low is not None and high is not None and high < low:
        raise ValueError(f"{filename}: row {row_number}: the high frequency precedes the low frequency")
    media_keys = list(_audacity_name_keys(filename))
    return _AudacityLabel(
        filename=filename,
        row_number=row_number,
        media_keys=media_keys,
        start=start,
        end=end,
        low=low,
        high=high,
        label=label or None,
    )


def _audio_media_by_label_key(
    session: Session,
    project_id: int,
    collection_id: int,
) -> dict[str, Media]:
    """Index the audio media of one project collection by their file-name keys."""
    media_by_key: dict[str, Media] = {}
    media_ids = session.exec(
        select(MediaCollection.media_id)
        .join(ProjectCollection, ProjectCollection.collection_id == MediaCollection.collection_id)
        .where(
            MediaCollection.collection_id == collection_id,
            ProjectCollection.project_id == project_id,
        )
    ).all()
    if not media_ids:
        return media_by_key
    media_rows = session.exec(
        select(Media).where(
            Media.media_id.in_(media_ids),
            Media.media_type == "audio",
        )
    ).all()
    for media in sorted(media_rows, key=lambda media: media.media_id):
        for key in _audacity_name_keys(media.filename) | _audacity_name_keys(media.name):
            media_by_key.setdefault(key, media)
    return media_by_key


def _audacity_label_bounds(item: _AudacityLabel, media: Media) -> tuple[float, float, float, float]:
    """Resolve annotation bounds; point labels get zero duration and unlabeled rows get the full band."""
    end = item.end if item.end is not None else item.start
    duration = media.audio_setting.duration_s if media.audio_setting else None
    if duration is not None and duration > 0:
        if item.start > duration + 1e-6:
            raise ValueError(f"{item.filename}: row {item.row_number}: the label starts after the end of the media")
        end = min(end, duration)
    if item.low is not None and item.high is not None:
        min_y, max_y = item.low, item.high
    else:
        sampling_rate_hz = media.audio_setting.sampling_rate_hz if media.audio_setting else None
        max_y = sampling_rate_hz / 2.0 if sampling_rate_hz and sampling_rate_hz > 0 else _DEFAULT_NYQUIST_HZ
        min_y, max_y = 0.0, max_y
    return item.start, end, min_y, max_y


def import_audacity_labels(
    session: Session,
    files: list[tuple[str, bytes]],
    user: User,
    project_id: int,
    collection_id: int,
    *,
    dry_run: bool,
) -> ImportResult:
    """
    Import Audacity label files as annotations.

    Each label file is matched to the audio media of the collection that shares
    its file name (ignoring the file extension). The label text becomes the
    annotation comments. Region labels use the start/end times, point labels
    (single timestamp) produce zero-duration annotations, and the optional
    low/high frequency columns define the frequency bounds (full audio band
    otherwise). Like the other tabular imports, validation is atomic: any
    failed row prevents any write.
    """
    report = ImportResult(delimiter="\t", dry_run=dry_run)
    parsed: list[_AudacityLabel] = []
    row_number = 0
    for filename, content in files:
        extension = Path(filename).suffix.lower().lstrip(".")
        if extension not in _AUDACITY_IMPORT_EXTENSIONS:
            report.global_errors.append(f"{filename}: unsupported file type; use Audacity label TXT or CSV files")
            continue
        try:
            text = _decode_import_content(content).strip()
        except HTTPException as exc:
            report.global_errors.append(f"{filename}: {exc.detail}")
            continue
        if not text:
            report.global_errors.append(f"{filename}: no labels found")
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            row_number += 1
            try:
                parsed.append(_parse_audacity_line(line, filename, row_number))
            except ValueError as exc:
                report.rows.append(ImportRowResult(row_number=row_number, status="failed", reason=str(exc)))

    media_by_key = _audio_media_by_label_key(session, project_id, collection_id) if parsed else {}
    results = {row.row_number: row for row in report.rows}
    accepted: list[tuple[Media, _AudacityLabel, float, float, float, float]] = []
    seen: set[tuple[int, float, float, float, float, str | None]] = set()
    for item in parsed:
        result = results.get(item.row_number)
        if result is None:
            result = ImportRowResult(row_number=item.row_number, status="succeeded")
            report.rows.append(result)
            results[item.row_number] = result
        media = next(
            (media_by_key[key] for key in item.media_keys if key in media_by_key),
            None,
        )
        if media is None:
            result.status = "failed"
            result.reason = (
                f"{item.filename}: no audio media named '{Path(item.filename).stem}' "
                "was found in the selected collection"
            )
            continue
        try:
            min_x, max_x, min_y, max_y = _audacity_label_bounds(item, media)
        except ValueError as exc:
            result.status = "failed"
            result.reason = str(exc)
            continue
        if item.label and len(item.label) > _MAX_ANNOTATION_COMMENTS_LENGTH:
            item.label = item.label[:_MAX_ANNOTATION_COMMENTS_LENGTH]
        comments = item.label
        dedup_key = (media.media_id, min_x, max_x, min_y, max_y, comments)
        if dedup_key in seen:
            result.status = "skipped"
            result.reason = "Duplicate label in file"
            continue
        seen.add(dedup_key)
        accepted.append((media, item, min_x, max_x, min_y, max_y))

    report.finalize()
    if report.failed or report.global_errors:
        if not dry_run:
            report.reject_candidates()
        return report
    if dry_run:
        return report.finalize()

    try:
        creation_date = datetime.now(UTC)
        session.add_all(
            Annotation(
                media_id=media.media_id,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                object_type=None,
                reference=False,
                comments=item.label,
                individual_num=1,
                creator_id=user.user_id,
                creator_type="user",
                creation_date=creation_date,
            )
            for media, item, min_x, max_x, min_y, max_y in accepted
        )
        session.commit()
    except Exception:
        session.rollback()
        report.global_errors.append("No data was written because the import transaction failed")
        report.reject_candidates()
        return report
    report.committed = True
    return report.finalize()


def import_reviews(
    session: Session,
    text: str,
    user: User,
    project_id: int,
    collection_id: int,
    *,
    dry_run: bool,
) -> ImportResult:
    report, rows = _parse_models(text, ReviewCreate, injected={"project_id": project_id})
    return _execute(
        session,
        report,
        rows,
        lambda item: (
            _require_media_scope(
                session,
                _annotation_media_id(session, item.annotation_id),
                project_id,
                collection_id,
            ),
            review_service.create_review(session, user, item, commit=False),
        ),
        dry_run=dry_run,
        key=lambda item: item.annotation_id,
        preflight=lambda item: (
            _require_media_scope(session, _annotation_media_id(session, item.annotation_id), project_id, collection_id),
            review_service.validate_review_create(session, user, item),
        ),
    )


def import_index_logs(
    session: Session,
    text: str,
    user: User,
    project_id: int,
    collection_id: int,
    *,
    dry_run: bool,
) -> ImportResult:
    report, rows = _parse_models(
        text,
        IndexLogCreateRequest,
        injected={"project_id": project_id},
        converters={
            "params": lambda value: _parse_json_cell(value, "params"),
            "results": lambda value: _parse_json_cell(value, "results"),
        },
    )
    return _execute(
        session,
        report,
        rows,
        lambda item: (
            _require_media_scope(
                session, item.media_id, project_id, collection_id
            ),
            analysis_service.analysis_service.save_acoustic_index_preview(
                session, item, user, commit=False
            ),
        ),
        dry_run=dry_run,
        key=lambda item: (item.media_id, item.index_id, item.version, json.dumps(item.results, sort_keys=True)),
        preflight=lambda item: (
            _require_media_scope(
                session, item.media_id, project_id, collection_id
            ),
            analysis_service.analysis_service.validate_acoustic_index_preview(
                session, item, user
            ),
        ),
    )


def import_tasks(
    session: Session,
    text: str,
    user: User,
    project_id: int,
    collection_id: int,
    *,
    dry_run: bool,
) -> ImportResult:
    report, rows = _parse_models(text, TaskImportRow)

    def write(item: SQLModel) -> None:
        task = TaskImportRow.model_validate(item)
        _require_media_scope(session, task.media_id, project_id, collection_id)
        task_service.assign_tasks(
            session,
            task.media_id,
            user,
            task.type,
            [{"user_id": task.assignee_id, "comment": task.comment}],
            [task.annotation_id] if task.annotation_id is not None else None,
            project_id=project_id,
            commit=False,
        )

    return _execute(
        session,
        report,
        rows,
        write,
        dry_run=dry_run,
        key=lambda item: (item.type, item.media_id, item.annotation_id, item.assignee_id),
        preflight=lambda item: (
            _require_media_scope(session, item.media_id, project_id, collection_id),
            task_service.validate_task_assignments(
                session, item.media_id, user, item.type,
                [{"user_id": item.assignee_id, "comment": item.comment}],
                [item.annotation_id] if item.annotation_id is not None else None,
            ),
        ),
    )


def import_users(
    session: Session,
    text: str,
    user: User,
    project_id: int,
    collection_id: int | None,
    *,
    dry_run: bool,
) -> ImportResult:
    report, rows = _parse_models(text, UserCreate)

    usernames: set[str] = set()
    emails: set[str] = set()
    results = {row.row_number: row for row in report.rows}
    filtered: list[tuple[int, SQLModel]] = []
    for row_number, item in rows:
        username = str(item.username).casefold()
        email = str(item.email).casefold()
        if username in usernames or email in emails:
            results[row_number].status = "skipped"
            results[row_number].reason = "Duplicate username or email in file"
            continue
        usernames.add(username)
        emails.add(email)
        filtered.append((row_number, item))
    report.finalize()

    return _execute(
        session,
        report,
        filtered,
        lambda item: user_service.create_user(
            session,
            user,
            item,
            project_id,
            collection_id,
            commit=False,
        ),
        dry_run=dry_run,
        preflight=lambda item: user_service.validate_user_create(
            session, user, item, project_id, collection_id
        ),
    )
