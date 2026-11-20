import pytest
from src.platform import Platform


@pytest.fixture
def plat(tmp_path):
    return Platform(db_path=tmp_path / "t.db", bundle_path=tmp_path / "none.joblib", dataset_dir=tmp_path / "ds",
                    backup_dir=tmp_path / "bk", reports_dir=tmp_path / "rep")
