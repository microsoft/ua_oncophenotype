import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CopyDirectoryHook:
    """Hook to copy a directory to the output directory. The contents of 'src' will be
    added to the contents of 'dst' if it exists, or 'dst' will be created if it does not
    exist."""

    src: Path
    dst: Path

    def __post_init__(self):
        self.src = Path(self.src)
        self.dst = Path(self.dst)

    def __call__(self):
        logger.info(f"Copying {self.src} to {self.dst}")
        shutil.copytree(self.src, self.dst, dirs_exist_ok=True)


@dataclass
class EnsureEmptyDirectoryHook:
    """Hook to ensure a directory is empty."""

    dir: Path

    def __post_init__(self):
        self.dir = Path(self.dir)

    def __call__(self):
        logger.info(f"Ensuring {self.dir} is created and empty")
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True, exist_ok=True)


@dataclass
class EnsureDirectoryExistsHook:
    """Hook to ensure a directory exists (create if it doesn't)"""

    dir: Path

    def __post_init__(self):
        self.dir = Path(self.dir)

    def __call__(self):
        logger.info(f"Ensuring {self.dir} exists")
        self.dir.mkdir(parents=True, exist_ok=True)
