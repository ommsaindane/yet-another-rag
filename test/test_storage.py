from pathlib import Path
from models.chunk import ChildChunk, ParentChunk
from storage.child_store import ChildStore
from storage.manifest import Manifest
from storage.parent_store import ParentStore


def test_manifest_incremental_diff(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    manifest_file = tmp_path / "manifest.json"

    doc_a = raw_dir / "a.md"
    doc_b = raw_dir / "b.txt"
    doc_a.write_text("Hello world", encoding="utf-8")
    doc_b.write_text("Second file", encoding="utf-8")

    manifest = Manifest(raw_directory=raw_dir, manifest_path=manifest_file)

    diff1 = manifest.get_diff()
    assert len(diff1.added) == 2
    assert len(diff1.modified) == 0
    assert len(diff1.removed) == 0
    manifest.save()

    assert not manifest.get_diff().has_changes

    doc_a.write_text("Hello updated world", encoding="utf-8")
    doc_b.unlink()

    diff2 = manifest.get_diff()
    assert len(diff2.added) == 0
    assert [p.name for p in diff2.modified] == ["a.md"]
    assert diff2.removed == ["b.txt"]


def test_parent_and_child_store_granular_deletion(tmp_path: Path):
    p_store = ParentStore(store_path=tmp_path / "parents.json")
    c_store = ChildStore(store_path=tmp_path / "children.json")

    p_store.add(
        [
            ParentChunk(id="d1_p0", document_id="d1", text="Doc 1 parent"),
            ParentChunk(id="d2_p0", document_id="d2", text="Doc 2 parent"),
        ]
    )
    c_store.add(
        [
            ChildChunk(id="d1_c0", parent_id="d1_p0", document_id="d1", text="Doc 1 child"),
            ChildChunk(id="d2_c0", parent_id="d2_p0", document_id="d2", text="Doc 2 child"),
        ]
    )

    assert p_store.delete_by_document_id("d1") == 1
    assert c_store.delete_by_document_id("d1") == 1
    assert p_store.get("d1_p0") is None
    assert p_store.get("d2_p0") is not None
    assert len(c_store.get()) == 1