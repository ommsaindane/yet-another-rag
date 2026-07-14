from hashlib import sha256
from pathlib import Path
import json


class Manifest:
    def __init__(
        self,
        raw_directory: str | Path,
        manifest_path: str | Path = "data/manifest.json",
    ):
        self.raw_directory = Path(raw_directory)
        self.manifest_path = Path(manifest_path)

    def documents_changed(self) -> bool:

        current = self._build_manifest()

        if not self.manifest_path.exists():
            return True

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            previous = json.load(f)

        return current != previous

    def save(self) -> None:

        manifest = self._build_manifest()

        self.manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.manifest_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                manifest,
                f,
                indent=4,
            )

    def _build_manifest(self) -> dict[str, str]:

        manifest: dict[str, str] = {}

        for file in sorted(self.raw_directory.glob("*")):
            if not file.is_file():
                continue

            manifest[file.name] = self._hash_file(file)

        return manifest

    @staticmethod
    def _hash_file(
        file: Path,
    ) -> str:

        hasher = sha256()

        with open(file, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

        return hasher.hexdigest()
