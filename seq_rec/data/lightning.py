from __future__ import annotations

import datetime
import functools
import math
import os
import pathlib
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pydantic
import torch
import torch.utils.data as torch_data
from lightning import LightningDataModule
from sentence_transformers import SentenceTransformer

from seq_rec.data.load import embed_example, merge_examples, nest_example, select_fields
from seq_rec.params import (
    BATCH_SIZE,
    DATA_DIR,
    ENCODER_NAME,
    ITEM_ID_COL,
    ITEM_TEXT_COL,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    MAX_SEQ_LEN,
    MOVIELENS_1M_URL,
    TARGET_COL,
    TOP_K,
    USER_ID_COL,
    USER_TEXT_COL,
    USERS_TABLE_NAME,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, TypeVar

    import lancedb
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import polars as pl

    T = TypeVar("T")


class FeaturesType(TypedDict):
    idx: int | torch.Tensor
    text: str | list[str]
    embedding: torch.Tensor


class BatchType(TypedDict):
    target: torch.Tensor
    user: dict[str, torch.Tensor]
    item: dict[str, torch.Tensor]
    neg_item: dict[str, torch.Tensor]


class FeaturesProcessor(pydantic.BaseModel):
    id_col: str
    text_col: str
    lance_table_name: str

    encoder_name: str = ENCODER_NAME
    batch_size: int = BATCH_SIZE

    data_dir: str = DATA_DIR
    lance_db_path: str = LANCE_DB_PATH

    @functools.cached_property
    def encoder(self) -> SentenceTransformer:
        return SentenceTransformer(self.encoder_name)

    @property
    def embedding_dim(self) -> int:
        return self.encoder.get_sentence_embedding_dimension()

    def embed(self, example: dict[str, Any]) -> npt.NDArray[np.float64]:
        return self.encoder.encode([example[self.text_col]])

    def process(self, example: dict[str, Any]) -> FeaturesType:
        import xxhash

        return {
            **example,
            "idx": xxhash.xxh32_intdigest(str(example[self.id_col])),
            "text": example[self.text_col],
            "inputs_embeds": self.embed(example),
        }

    def collate(self, batch: list[FeaturesType]) -> FeaturesType:
        from seq_rec.data.load import torch_collate

        return torch_collate.default_collate(batch)

    @property
    def data_path(self) -> str:
        raise NotImplementedError

    def get_data(
        self, subset: str, cycle: int = 1
    ) -> torch_data.IterDataPipe[FeaturesType]:
        import pyarrow.dataset as ds

        from seq_rec.data.load import ParquetDictLoaderIterDataPipe

        valid_subset = {"train", "val", "test", "predict"}
        if subset not in valid_subset:
            msg = f"`{subset}` is not one of `{valid_subset}`"
            raise ValueError(msg)

        filter_expr = ds.field(f"is_{subset}")
        return (
            ParquetDictLoaderIterDataPipe([self.data_path], filter_expr=filter_expr)
            .cycle(count=cycle)
            .shuffle()  # devskim: ignore DS148264
            .sharding_filter()
        )

    def get_processed_data(
        self, subset: str, cycle: int = 1
    ) -> torch_data.IterDataPipe[FeaturesType]:
        return self.get_data(subset, cycle=cycle).map(self.process)

    def get_batch_data(
        self, subset: str, cycle: int | None = 1
    ) -> torch_data.IterDataPipe[FeaturesType]:
        fields = ["idx", "inputs_embeds"]
        return (
            self.get_processed_data(subset, cycle=cycle)
            .map(functools.partial(select_fields, fields=fields))
            .batch(self.batch_size)
            .collate(collate_fn=self.collate)
        )

    @property
    def lance_db(self) -> lancedb.DBConnection:
        import lancedb

        return lancedb.connect(self.lance_db_path)

    @property
    def lance_table(self) -> lancedb.table.Table:
        return self.lance_db.open_table(self.lance_table_name)

    def get_id(self, id_val: int | None) -> dict[str, Any]:
        if id_val is None:
            return {}
        result = self.lance_table.search().where(f"{self.id_col} = {id_val}").to_list()
        if len(result) == 0:
            return {}
        return result[0]


class ItemsProcessor(FeaturesProcessor):
    id_col: str = ITEM_ID_COL
    text_col: str = ITEM_TEXT_COL
    lance_table_name: str = ITEMS_TABLE_NAME

    num_partitions: int | None = None
    num_sub_vectors: int | None = None
    num_probes: int = 8
    refine_factor: int = 4

    @property
    def data_path(self) -> str:
        return pathlib.Path(self.data_dir, "ml-1m", "movies.parquet").as_posix()

    def get_index(
        self, model: torch.nn.Module, subset: str = "predict"
    ) -> lancedb.table.Table:
        import pyarrow as pa

        fields = [self.id_col, self.text_col, "embedding"]
        dp = (
            self.get_processed_data(subset)
            .map(functools.partial(torch.inference_mode(embed_example), model=model))
            .map(functools.partial(select_fields, fields=fields))
            .batch(self.batch_size)
        )

        batch = next(iter(dp))
        num_items = len(dp) * len(batch)
        example = batch[0]
        (embedding_dim,) = example["embedding"].shape

        # rule of thumb: nlist ~= 4 * sqrt(n_vectors)
        num_partitions = self.num_partitions or 2 ** int(math.log2(num_items) / 2)
        num_sub_vectors = self.num_sub_vectors or embedding_dim // 8

        if torch.cuda.is_available():
            accelerator = "cuda"
        elif torch.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = None

        schema = pa.RecordBatch.from_pylist(batch).schema
        schema = schema.set(
            schema.get_field_index("embedding"),
            pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
        )

        table = self.lance_db.create_table(
            self.lance_table_name,
            data=iter(dp.map(pa.RecordBatch.from_pylist)),
            schema=schema,
            mode="overwrite",
        )
        table.create_scalar_index(self.id_col)
        table.create_index(
            vector_column_name="embedding",
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            index_type="IVF_HNSW_PQ",
            accelerator=accelerator,
        )
        table.optimize(
            cleanup_older_than=datetime.timedelta(days=0),
            delete_unverified=True,
            retrain=True,
        )
        return table

    def search(
        self,
        embedding: npt.NDArray[np.float64],
        exclude_item_ids: list[int] | None = None,
        top_k: int = TOP_K,
    ) -> pd.DataFrame:
        if self.lance_table is None:
            msg = "`index` must be intialised first"
            raise ValueError(msg)

        exclude_item_ids = exclude_item_ids or [0]
        exclude_filter = ", ".join(f"{item}" for item in exclude_item_ids)
        exclude_filter = f"{self.id_col} NOT IN ({exclude_filter})"
        return (
            self.lance_table.search(embedding)
            .where(exclude_filter, prefilter=True)
            .nprobes(self.num_probes)
            .refine_factor(self.refine_factor)
            .limit(top_k)
            .to_pandas()
            .assign(score=lambda df: 1 - df["_distance"])
            .drop(columns="_distance")
        )


class UsersProcessor(FeaturesProcessor):
    id_col: str = USER_ID_COL
    text_col: str = USER_TEXT_COL
    lance_table_name: str = USERS_TABLE_NAME

    items_processor: ItemsProcessor
    max_seq_len: int = MAX_SEQ_LEN

    def embed(self, example: dict[str, Any]) -> npt.NDArray[np.float64]:
        history_text = (
            item[self.items_processor.text_col] for item in example.get("history", [])
        )
        history_text = list(reversed([*history_text, example[self.text_col]]))
        history_text = history_text[: self.max_seq_len]
        return self.encoder.encode(history_text)

    @property
    def data_path(self) -> str:
        return pathlib.Path(self.data_dir, "ml-1m", "users.parquet").as_posix()

    def get_index(self, subset: str = "predict") -> lancedb.table.Table:
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq

        columns = [self.id_col, self.text_col, "history", "target"]
        filters = ds.field(f"is_{subset}")
        pa_table = pq.read_table(self.data_path, columns=columns, filters=filters)

        table = self.lance_db.create_table(
            self.lance_table_name, data=pa_table, mode="overwrite"
        )
        table.create_scalar_index(self.id_col)
        table.optimize(
            cleanup_older_than=datetime.timedelta(days=0),
            delete_unverified=True,
            retrain=True,
        )
        return table

    def get_activity(self, id_val: int | None, activity_name: str) -> dict[int, int]:
        activity = self.get_id(id_val).get(activity_name, {})
        return {item[ITEM_ID_COL]: item[TARGET_COL] for item in activity}


class InteractionsProcessor(pydantic.BaseModel):
    users_processor: UsersProcessor
    items_processor: ItemsProcessor

    target_col: str = TARGET_COL
    batch_size: int = BATCH_SIZE
    data_dir: str = DATA_DIR

    def process(self, example: dict[str, Any]) -> BatchType:
        fields = ["idx", "inputs_embeds"]
        user_features = select_fields(
            self.users_processor.process(example), fields=fields
        )
        item_features = select_fields(
            self.items_processor.process(example), fields=fields
        )
        target = example[self.target_col]
        return {"target": target, "user": user_features, "item": item_features}

    def collate(self, batch: list[BatchType]) -> BatchType:
        import torch.utils.data._utils.collate as torch_collate

        target = torch_collate.default_collate([example["target"] for example in batch])
        user = self.users_processor.collate([example["user"] for example in batch])
        item = self.items_processor.collate([example["item"] for example in batch])
        neg_item = self.items_processor.collate(
            [example["neg_item"] for example in batch]
        )
        return {"target": target, "user": user, "item": item, "neg_item": neg_item}

    def get_processed_data(self, subset: str) -> torch_data.IterDataPipe[BatchType]:
        import pyarrow.dataset as ds

        from seq_rec.data.load import ParquetDictLoaderIterDataPipe

        valid_subset = {"train", "val", "test", "predict"}
        if subset not in valid_subset:
            msg = f"`{subset}` is not one of `{valid_subset}`"
            raise ValueError(msg)

        data_path = pathlib.Path(self.data_dir, "ml-1m", "ratings.parquet").as_posix()
        filter_expr = ds.field(f"is_{subset}")
        fields = ["idx", "inputs_embeds"]
        neg_item_dp = (
            self.items_processor.get_processed_data(subset, cycle=None)
            .map(functools.partial(select_fields, fields=fields))
            .map(functools.partial(nest_example, key="neg_item"))
        )
        return (
            ParquetDictLoaderIterDataPipe([data_path], filter_expr=filter_expr)
            .shuffle()  # devskim: ignore DS148264
            .sharding_filter()
            .map(self.process)
            .zip(neg_item_dp)
            .map(merge_examples)
        )

    def get_batch_data(self, subset: str) -> torch_data.IterDataPipe[FeaturesType]:
        return (
            self.get_processed_data(subset)
            .batch(self.batch_size)
            .collate(collate_fn=self.collate)
        )


class SeqRecDataModule(LightningDataModule):
    def __init__(  # noqa: PLR0913
        self,
        data_dir: str = DATA_DIR,
        batch_size: int = BATCH_SIZE,
        num_partitions: int | None = None,
        num_sub_vectors: int | None = None,
        num_probes: int = 8,
        refine_factor: int = 4,
        num_workers: int | None = 1,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.items_processor = ItemsProcessor(
            batch_size=batch_size,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            num_probes=num_probes,
            refine_factor=refine_factor,
            data_dir=data_dir,
        )
        self.users_processor = UsersProcessor(
            items_processor=self.items_processor,
            batch_size=batch_size,
            data_dir=data_dir,
        )
        self.interactions_processor = InteractionsProcessor(
            users_processor=self.users_processor,
            items_processor=self.items_processor,
            batch_size=batch_size,
            data_dir=data_dir,
        )

    def prepare_data(self, *, overwrite: bool = False) -> pl.LazyFrame:
        from filelock import FileLock

        from seq_rec.data.prepare import download_unpack_data, prepare_movielens

        data_dir = self.hparams.data_dir
        with FileLock(f"{data_dir}.lock"):
            download_unpack_data(MOVIELENS_1M_URL, data_dir, overwrite=overwrite)
            return prepare_movielens(data_dir, overwrite=overwrite)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_data = self.interactions_processor.get_processed_data("train")

        if stage in {"fit", "validate"}:
            self.val_data = self.users_processor.get_processed_data("val")

        if stage == "test":
            self.test_data = self.users_processor.get_processed_data("test")

        if stage == "predict":
            self.predict_data = self.users_processor.get_processed_data("predict")

    def get_dataloader(
        self,
        dataset: torch_data.Dataset[T],
        *,
        batch_size: int | None = None,
        collate_fn: Callable[[list[T]], T] | None = None,
        shuffle: bool = False,
    ) -> torch_data.DataLoader[T]:
        num_workers = self.hparams.get("num_workers")

        if num_workers is None:
            num_workers = os.cpu_count() or 1

        multiprocessing_context = "spawn" if num_workers > 0 else None

        return torch_data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            multiprocessing_context=multiprocessing_context,
            persistent_workers=num_workers > 0,
        )

    def train_dataloader(self) -> torch_data.DataLoader[BatchType]:
        batch_size = self.hparams.get("batch_size")
        return self.get_dataloader(
            self.train_data,
            batch_size=batch_size,
            collate_fn=self.interactions_processor.collate,
            shuffle=True,
        )

    def val_dataloader(self) -> torch_data.DataLoader[FeaturesType]:
        return self.get_dataloader(self.val_data)

    def test_dataloader(self) -> torch_data.DataLoader[FeaturesType]:
        return self.get_dataloader(self.test_data)

    def predict_dataloader(self) -> torch_data.DataLoader[FeaturesType]:
        return self.get_dataloader(self.predict_data)


if __name__ == "__main__":
    import rich

    dm = SeqRecDataModule()
    dm.prepare_data().head().collect().glimpse()
    dm.setup("fit")

    dataloaders = [
        dm.items_processor.get_batch_data("train"),
        dm.users_processor.get_batch_data("train"),
        dm.interactions_processor.get_processed_data("train"),
        dm.interactions_processor.get_batch_data("train"),
        dm.train_dataloader(),
        dm.val_dataloader(),
    ]
    for dataloader in dataloaders:
        batch = next(iter(dataloader))
        rich.print(batch)
        shapes = {
            key: value.shape
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        rich.print(shapes)

    dm.users_processor.get_index().search().to_polars().glimpse()
    rich.print(dm.users_processor.get_id(1))
    rich.print(dm.users_processor.get_activity(1, "history"))
    rich.print(dm.users_processor.get_activity(1, "target"))

    dm.items_processor.get_index(torch.squeeze).search().to_polars().glimpse()
    rich.print(dm.items_processor.get_id(1))
    query_vector = torch.rand(  # devskim: ignore DS148264
        dm.items_processor.embedding_dim
    ).numpy()
    rich.print(dm.items_processor.search(query_vector, top_k=5))
