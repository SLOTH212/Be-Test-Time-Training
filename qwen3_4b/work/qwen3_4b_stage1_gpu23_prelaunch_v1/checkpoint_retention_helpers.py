"""Process-crash-safe A/B publication, preserving previous until commit."""
def _validate_committed(root):
    latest_path = root / "latest.json"
    if not latest_path.is_file():
        raise RuntimeError("DCP_UNCOMMITTED_ARTIFACTS_NO_LATEST")
    latest = json.loads(latest_path.read_text())
    if latest.get("slot") not in ("slot_A", "slot_B"):
        raise RuntimeError("DCP_INVALID_SLOT")
    final = root / latest["slot"]
    manifest_path = final / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if _digest(manifest_path) != latest["manifest_sha256"] or not manifest.get("complete"):
        raise RuntimeError("DCP_MANIFEST_INVALID")
    _validate_checkpoint_content(final, manifest)
    if manifest["progress"] != latest["progress"]:
        raise RuntimeError("DCP_PROGRESS_POINTER_MISMATCH")
    return latest

def _validate_checkpoint_content(directory, manifest):
    if not manifest.get("files"):
        raise RuntimeError("DCP_EMPTY_FILE_MANIFEST")
    for name, digest in manifest["files"].items():
        candidate = (directory / name).resolve()
        if not candidate.is_relative_to(directory.resolve()) or _digest(candidate) != digest:
            raise RuntimeError("DCP_FILE_HASH_MISMATCH " + name)

def _remove_retention_artifact(path, root):
    if path.parent.resolve() != root.resolve() or path.is_symlink():
        raise RuntimeError("DCP_UNSAFE_RETENTION_PATH")
    if path.exists():
        shutil.rmtree(path)
        fsync_dir(root)

def _recover_publication(root):
    backups = [root / (slot + ".previous") for slot in ("slot_A", "slot_B")
               if (root / (slot + ".previous")).exists()]
    if not (root / "latest.json").exists():
        if any((root / name).exists() for name in
               ("slot_A", "slot_B", "slot_A.previous", "slot_B.previous",
                "slot_A.partial", "slot_B.partial")):
            raise RuntimeError("DCP_UNCOMMITTED_ARTIFACTS_NO_LATEST")
        return
    latest = _validate_committed(root)
    if not backups:
        return
    if len(backups) != 1:
        raise RuntimeError("DCP_AMBIGUOUS_PUBLICATION_BACKUPS")
    backup = backups[0]
    slot = backup.name.removesuffix(".previous")
    final = root / slot
    if latest["slot"] == slot:
        _remove_retention_artifact(backup, root)
    else:
        # Before commit, backup is the old previous known-good.
        # Recovery can be repeated after interruption of either operation.
        _remove_retention_artifact(final, root)
        os.replace(backup, final)
        fsync_dir(root)

def _publish_checkpoint(partial, final, progress):
    root = final.parent
    backup = root / (final.name + ".previous")
    if backup.exists():
        raise RuntimeError("DCP_UNRECOVERED_PUBLICATION")
    manifest = json.loads((partial / "manifest.json").read_text())
    if not manifest.get("complete") or manifest["slot"] != final.name:
        raise RuntimeError("DCP_PARTIAL_MANIFEST_INVALID")
    if manifest["progress"] != progress:
        raise RuntimeError("DCP_PARTIAL_PROGRESS_MISMATCH")
    _validate_checkpoint_content(partial, manifest)
    if final.exists():
        os.replace(final, backup)
        fsync_dir(root)
    os.replace(partial, final)
    fsync_dir(root)
    atomic_json(root / "latest.json", {
        "slot": final.name, "progress": progress,
        "manifest_sha256": _digest(final / "manifest.json")})
    _remove_retention_artifact(backup, root)
