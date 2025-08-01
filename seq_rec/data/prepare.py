from __future__ import annotations

import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from loguru import logger

from seq_rec.params import DATA_DIR, MOVIELENS_1M_URL

###
# download data
###


def download_data(
    url: str = MOVIELENS_1M_URL, dest_dir: str = DATA_DIR, *, overwrite: bool = False
) -> pathlib.Path:
    import requests

    # prepare destination
    dest = pathlib.Path(dest_dir, pathlib.Path(url).name)

    # download zip
    if not dest.exists() or overwrite:
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading data: {}", url)
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        with dest.open("wb") as f:
            shutil.copyfileobj(response.raw, f)

    logger.info("data downloaded: {}", dest)
    return dest


def unpack_data(
    archive_file: str | pathlib.Path, *, overwrite: bool = False
) -> list[str]:
    archive_file = pathlib.Path(archive_file)
    dest_dir = archive_file.parent / archive_file.stem

    if not dest_dir.exists() or overwrite:
        shutil.unpack_archive(archive_file, dest_dir.parent)

    unpacked_files = [file.name for file in dest_dir.iterdir()]
    logger.info("data unpacked: {}", unpacked_files)
    return unpacked_files


def download_unpack_data(
    url: str = MOVIELENS_1M_URL, dest_dir: str = DATA_DIR, *, overwrite: bool = False
) -> list[str]:
    archive_file = download_data(url=url, dest_dir=dest_dir, overwrite=overwrite)
    return unpack_data(archive_file, overwrite=overwrite)


###
# load data
###


def load_movies(src_dir: str = DATA_DIR) -> pl.LazyFrame:
    import pandas as pd

    movies_dat = pathlib.Path(src_dir, "ml-1m", "movies.dat")
    dtype = {"movie_id": "int32", "title": "str", "genres": "str"}
    movies = (
        pd.read_csv(
            movies_dat,
            sep="::",
            header=None,
            names=list(dtype.keys()),
            dtype=dtype,
            engine="python",
            encoding="iso-8859-1",
        )
        .pipe(pl.from_pandas)
        .with_columns(genres=pl.col("genres").str.split("|"))
        .with_columns(movie_text=pl.struct("title", "genres").struct.json_encode())
        .drop("title", "genres")
    )
    logger.info("movies loaded: {}, shape: {}", movies_dat, movies.shape)

    return movies.lazy()


def load_users(src_dir: str = DATA_DIR) -> pl.LazyFrame:
    import pandas as pd

    users_dat = pathlib.Path(src_dir, "ml-1m", "users.dat")
    dtype = {
        "user_id": "int32",
        "gender": "str",
        "age": "int32",
        "occupation": "int32",
        "zipcode": "str",
    }

    users = (
        pd.read_csv(
            users_dat,
            sep="::",
            header=None,
            names=list(dtype.keys()),
            dtype=dtype,
            engine="python",
        )
        .pipe(pl.from_pandas)
        .with_columns(
            user_text=pl.struct(
                "gender", "age", "occupation", "zipcode"
            ).struct.json_encode()
        )
        .drop("gender", "age", "occupation", "zipcode")
    )
    logger.info("users loaded: {}, shape: {}", users_dat, users.shape)
    return users.lazy()


def load_ratings(src_dir: str = DATA_DIR) -> pl.LazyFrame:
    import pandas as pd

    ratings_dat = pathlib.Path(src_dir, "ml-1m", "ratings.dat")
    dtype = {
        "user_id": "int32",
        "movie_id": "int32",
        "rating": "int32",
        "timestamp": "int32",
    }
    ratings = (
        pd.read_csv(
            ratings_dat,
            sep="::",
            header=None,
            names=list(dtype.keys()),
            dtype=dtype,
            engine="python",
        )
        .pipe(pl.from_pandas)
        .with_columns(datetime=pl.from_epoch("timestamp"))
    )
    logger.info("ratings loaded: {}, shape: {}", ratings_dat, ratings.shape)
    return ratings.lazy()


###
# process data
###


def train_test_split(
    ratings: pl.LazyFrame,
    *,
    group_col: str = "user_id",
    order_col: str = "datetime",
    train_prop: float = 0.8,
    val_prop: float = 0.2,
) -> pl.LazyFrame:
    ratings = (
        ratings.lazy()
        .with_columns(
            p=((pl.col(order_col).rank("min") - 1) / pl.count(order_col)).over(
                group_col
            ),
        )
        # first train_prop will be train set
        .with_columns(is_train=pl.col("p") < train_prop)
        .drop("p")
    )
    users_split = (
        ratings.filter(~pl.col("is_train"))
        .group_by(group_col)
        .len()
        .with_columns(p=((pl.col("len").rank("min") - 1) / pl.count("len")))
        # largest val_prop by count will be val set
        .with_columns(is_val=pl.col("p") >= 1 - val_prop)
        .drop("len", "p")
    )
    return ratings.join(
        users_split, on=group_col, how="left", validate="m:1"
    ).with_columns(
        is_val=~pl.col("is_train") & pl.col("is_val"),
        is_test=~pl.col("is_train") & ~pl.col("is_val"),
        is_predict=True,
    )


def process_ratings(
    ratings: pl.LazyFrame,
    users: pl.LazyFrame,
    movies: pl.LazyFrame,
    *,
    src_dir: str = DATA_DIR,
    overwrite: bool = False,
) -> pl.LazyFrame:
    ratings_parquet = pathlib.Path(src_dir, "ml-1m", "ratings.parquet")
    if ratings_parquet.exists() and not overwrite:
        ratings_processed = pl.scan_parquet(ratings_parquet)
        logger.info("ratings loaded: {}", ratings_parquet)
        return ratings_processed

    ratings_merged = (
        ratings.lazy()
        .join(movies.lazy(), on="movie_id", how="left", validate="m:1")
        .join(users.lazy(), on="user_id", how="left", validate="m:1")
        .sort(["user_id", "datetime"])
    )

    with ThreadPoolExecutor() as executor:
        for _, df in ratings_merged.collect().group_by("user_id"):
            executor.submit(gather_history, ratings=df.lazy(), path=ratings_parquet)

    ratings_processed = pl.scan_parquet(ratings_parquet)
    n_row = ratings_processed.select(pl.len()).collect().item()
    n_col = ratings_processed.collect_schema().len()
    logger.info("ratings saved: {}, shape: {}", ratings_parquet, (n_row, n_col))
    return ratings_processed


def gather_history(ratings: pl.LazyFrame, *, path: pathlib.Path) -> pl.LazyFrame:
    ratings_history = (
        ratings.rolling("datetime", period="4w", closed="none", group_by="user_id")
        .agg(history=pl.struct("datetime", "rating", "movie_id", "movie_text"))
        .unique(["user_id", "datetime"])
    )
    ratings_target = ratings.group_by("user_id", "is_train").agg(
        target=pl.struct("rating", "movie_id")
    )
    ratings_history = ratings.join(
        ratings_history, on=["user_id", "datetime"], validate="m:1"
    ).join(ratings_target, on=["user_id", "is_train"], validate="m:1")
    ratings_history.collect().write_parquet(path, partition_by="user_id")
    return ratings_history


def process_movies(
    movies: pl.LazyFrame,
    ratings: pl.LazyFrame,
    *,
    src_dir: str = DATA_DIR,
    overwrite: bool = False,
) -> pl.LazyFrame:
    movies_parquet = pathlib.Path(src_dir, "ml-1m", "movies.parquet")
    if movies_parquet.exists() and not overwrite:
        movies_processed = pl.scan_parquet(movies_parquet)
        logger.info("movies loaded: {}", movies_parquet)
        return movies_processed

    movies_train = ratings.lazy().group_by("movie_id").agg(pl.any("is_train"))
    movies_processed = (
        movies.lazy()
        .join(movies_train, on="movie_id", how="left", validate="1:1")
        .with_columns(is_val=True, is_test=True, is_predict=True)
        .collect()
    )

    movies_processed.write_parquet(movies_parquet)
    logger.info("movies saved: {}, shape: {}", movies_parquet, movies_processed.shape)
    return pl.scan_parquet(movies_parquet)


def process_users(
    users: pl.LazyFrame,
    ratings: pl.LazyFrame,
    *,
    src_dir: str = DATA_DIR,
    overwrite: bool = False,
) -> pl.LazyFrame:
    users_parquet = pathlib.Path(src_dir, "ml-1m", "users.parquet")
    if users_parquet.exists() and not overwrite:
        users_procesed = pl.scan_parquet(users_parquet)
        logger.info("users loaded: {}", users_parquet)
        return users_procesed

    users_interactions = (
        ratings.lazy()
        .group_by("user_id")
        .agg(
            history=pl.struct("datetime", "rating", "movie_id", "movie_text").filter(
                "is_train"
            ),
            target=pl.struct("datetime", "rating", "movie_id", "movie_text").filter(
                ~pl.col("is_train")
            ),
            is_train=pl.any("is_train"),
            is_val=pl.any("is_val"),
            is_test=pl.any("is_test"),
            is_predict=pl.any("is_predict"),
        )
        .with_columns(
            history=pl.col("history").list.sort(),
            target=pl.col("target").list.sort(),
        )
    )
    users_procesed = (
        users.lazy()
        .join(users_interactions, on="user_id", how="left", validate="1:1")
        .collect()
    )

    users_procesed.write_parquet(users_parquet)
    logger.info("users saved: {}, shape: {}", users_parquet, users_procesed.shape)
    return pl.scan_parquet(users_parquet)


def prepare_movielens(
    src_dir: str = DATA_DIR, *, overwrite: bool = False
) -> pl.LazyFrame:
    movies = load_movies(src_dir)
    users = load_users(src_dir)
    ratings = load_ratings(src_dir).pipe(train_test_split)

    ratings = process_ratings(
        ratings, users, movies, src_dir=src_dir, overwrite=overwrite
    )
    movies = process_movies(movies, ratings, src_dir=src_dir, overwrite=overwrite)
    users = process_users(users, ratings, src_dir=src_dir, overwrite=overwrite)
    return ratings


def main(data_dir: str = DATA_DIR, *, overwrite: bool = True) -> None:
    download_unpack_data(overwrite=overwrite)
    with pl.StringCache():
        prepare_movielens(data_dir, overwrite=overwrite).head().collect().glimpse()


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
