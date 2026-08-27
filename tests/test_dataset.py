from __future__ import annotations

from pathlib import Path

import pytest

from bim_agent.dataset import DatasetError, ProjectDataset


def test_discovers_and_traverses_three_files(sample_data: Path) -> None:
    dataset = ProjectDataset(sample_data)
    assert len(dataset.records) == 22
    assert len(dataset.physical_records) == 8
    assert {record.category for record in dataset.physical_records} == {
        "Lighting Devices", "Electrical Fixtures", "Electrical Equipment", "Pipes"
    }
    assert len(dataset.source_manifest()) == 3


def test_sqlite_metadata_is_auditable_but_not_physical(sample_data: Path) -> None:
    dataset = ProjectDataset(sample_data)
    matches = dataset.search_sqlite_metadata(["switch"])
    assert matches[0]["entity_id"] == 9001
    assert matches[0]["in_tree"] is False


def test_rejects_incomplete_contract(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="exactly one"):
        ProjectDataset(tmp_path)
