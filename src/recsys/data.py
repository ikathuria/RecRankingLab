"""Download MovieLens 25M and load it into DuckDB."""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

import duckdb

URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
MD5 = "6b51fb2759a8657d3bfcbfc42b592ada"  # published at <URL>.md5
RATINGS_ROWS = 25_000_095
RAW_DIR = Path("data/raw")
DB_PATH = Path("data/movielens.duckdb")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path = RAW_DIR) -> Path:
    """Fetch and extract the dataset. Idempotent: a verified zip is not re-downloaded."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-25m.zip"
    if not (zip_path.exists() and _md5(zip_path) == MD5):
        urllib.request.urlretrieve(URL, zip_path)
        if (got := _md5(zip_path)) != MD5:
            zip_path.unlink()
            raise RuntimeError(f"checksum mismatch: expected {MD5}, got {got}")
    out = raw_dir / "ml-25m"
    if not (out / "ratings.csv").exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw_dir)
    return out


def build_db(csv_dir: Path, db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Load ratings, movies and tags into DuckDB (timestamps stay as unix seconds)."""
    con = duckdb.connect(str(db_path))
    con.execute(
        f"""
        CREATE OR REPLACE TABLE ratings AS
        SELECT userId AS user_id, movieId AS movie_id, rating, "timestamp" AS ts
        FROM read_csv('{csv_dir / "ratings.csv"}', header=true,
                      columns={{'userId':'INTEGER','movieId':'INTEGER','rating':'DOUBLE',
                               'timestamp':'BIGINT'}});
        CREATE OR REPLACE TABLE movies AS
        SELECT movieId AS movie_id, title, genres
        FROM read_csv('{csv_dir / "movies.csv"}', header=true);
        CREATE OR REPLACE TABLE tags AS
        SELECT userId AS user_id, movieId AS movie_id, tag, "timestamp" AS ts
        FROM read_csv('{csv_dir / "tags.csv"}', header=true,
                      columns={{'userId':'INTEGER','movieId':'INTEGER','tag':'VARCHAR',
                               'timestamp':'BIGINT'}});
        """
    )
    return con


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("download", "all"):
        csv_dir = download()
        print(f"data ready: {csv_dir}")
    if cmd in ("build", "all"):
        con = build_db(RAW_DIR / "ml-25m")
        n = con.execute("SELECT count(*) FROM ratings").fetchone()[0]
        print(f"ratings: {n}")
        if n != RATINGS_ROWS:
            raise SystemExit(f"expected {RATINGS_ROWS} ratings, got {n}")


if __name__ == "__main__":
    main()
