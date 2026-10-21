import duckdb

from recsys.data import build_db


def test_build_db_tables(tmp_path):
    (tmp_path / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n1,10,4.0,100\n1,11,3.5,200\n2,10,5.0,300\n"
    )
    (tmp_path / "movies.csv").write_text(
        "movieId,title,genres\n10,A (2000),Drama|Comedy\n11,B,Action\n"
    )
    (tmp_path / "tags.csv").write_text("userId,movieId,tag,timestamp\n1,10,nice,150\n")
    con = build_db(tmp_path, tmp_path / "t.duckdb")
    assert con.execute("SELECT count(*) FROM ratings").fetchone()[0] == 3
    assert con.execute("SELECT count(*) FROM movies").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM tags").fetchone()[0] == 1
    assert isinstance(con, duckdb.DuckDBPyConnection)
