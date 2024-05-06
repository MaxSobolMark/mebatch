from gcs_mutex_lock import gcs_lock


class GCSFileLock:
    def __init__(self, file_path: str):
        self.file_path = file_path
        assert file_path.startswith("gs://"), "The file path must start with gs://."

    def __enter__(self):
        acquired = gcs_lock.wait_for_lock_expo(self.file_path)
        assert acquired, f"Could not acquire lock for {self.file_path}."

    def __exit__(self, exc_type, exc_val, exc_tb):
        gcs_lock.unlock(self.file_path)
