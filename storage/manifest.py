from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json


@dataclass(slots=True)
class ManifestDiff:
    added: list[Path]
    modified: list[Path]
    removed: list[str]  # Relative POSIX paths of deleted files

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.removed)

    @property
    def files_to_ingest(self) -> list[Path]:
        return self.added + self.modified

    @property
    def rel_paths_to_remove(self) -> list[str]:
        """Relative paths whose old chunks/vectors must be purged before re-indexing."""
        return self.removed


class Manifest:
    def __init__(
        self,
        raw_directory: str | Path = "data/raw",
        manifest_path: str | Path = "data/manifest.json",
        supported_extensions: set[str] | None = None,
    ):
        self.raw_directory = Path(raw_directory).resolve()
        self.manifest_path = Path(manifest_path)
        self.supported_extensions = supported_extensions or {
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".md",
            ".txt",
            ".html",
        }

    @staticmethod
    def compute_document_id(relative_path: str) -> str:
        """
        Generates a deterministic 24-character hex ID from a normalized relative path.
        """
        normalized = relative_path.replace("\\", "/").strip("/")
        return sha256(normalized.encode("utf-8")).hexdigest()[:24]

    def get_relative_path(self, file_path: Path) -> str:
        return file_path.resolve().relative_to(self.raw_directory).as_posix()

    def get_document_id_for_path(self, file_path: Path) -> str:
        rel_path = self.get_relative_path(file_path)
        return self.compute_document_id(rel_path)

    def get_diff(self) -> ManifestDiff:
        current = self._build_manifest()
        previous = self._load_previous()

        added: list[Path] = []
        modified: list[Path] = []
        removed: list[str] = []

        for rel_path, file_hash in current.items():
            full_path = self.raw_directory / rel_path
            if rel_path not in previous:
                added.append(full_path)
            elif previous[rel_path] != file_hash:
                modified.append(full_path)

        for rel_path in previous:
            if rel_path not in current:
                removed.append(rel_path)

        return ManifestDiff(
            added=sorted(added),
            modified=sorted(modified),
            removed=sorted(removed),
        )

    def documents_changed(self) -> bool:
        return self.get_diff().has_changes

    def save(self) -> None:
        manifest = self._build_manifest()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, sort_keys=True)

    def _load_previous(self) -> dict[str, str]:
        if not self.manifest_path.exists():
            return {}
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _build_manifest(self) -> dict[str, str]:
        if not self.raw_directory.exists():
            return {}

        manifest: dict[str, str] = {}
        for file in sorted(self.raw_directory.rglob("*")):
            if not file.is_file():
                continue
            if file.suffix.lower() not in self.supported_extensions:
                continue
            rel_path = self.get_relative_path(file)
            manifest[rel_path] = self._hash_file(file)

        return manifest

    @staticmethod
    def _hash_file(file: Path) -> str:
        hasher = sha256()
        with open(file, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()