import argparse
import abc
from typing import Optional, List, Tuple
from logging import Logger, StreamHandler, INFO, Formatter
import tempfile
import os

import pandas as pd
from google.cloud import storage, bigquery
from utils import reduce_mem_usage
from google.cloud import bigquery_storage_v1beta1
from io import BytesIO


GCS_BUCKET_NAME = "kaggle-riiid-w"
PROJECT_ID = "wantedly-individual-shu"


class BaseFeature(abc.ABC):
    save_memory: bool = True

    def __init__(self, debugging: bool = False, **kwargs) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        self.debugging = debugging
        self._logger = Logger(self.__class__.__name__)
        handler = StreamHandler()
        fmt = Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(fmt)
        handler.setLevel(INFO)
        self._logger.addHandler(handler)

        self.GCS_BUCKET_NAME = GCS_BUCKET_NAME
        self.PROJECT_ID = PROJECT_ID

        self.train_table = f"`{PROJECT_ID}.riiid.train`"

    @abc.abstractmethod
    def import_columns(self) -> List[str]:
        """この特徴量を作るのに必要なカラムを指定する
        """
        ...

    @abc.abstractmethod
    def make_features(
        self, df_train_input: pd.DataFrame,
    ) -> pd.DataFrame:
        """BigQuery から取得した生データの DataFrame を特徴量に変換する
        """
        ...

    @classmethod
    def add_feature_specific_arguments(cls, parser: argparse.ArgumentParser):
        return

    @classmethod
    def main(cls):
        import logging

        logging.basicConfig(level=logging.INFO)
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action="store_true")
        cls.add_feature_specific_arguments(parser)
        args = parser.parse_args()
        instance = cls(debugging=args.debug, **vars(args))
        instance.run()

    def run(self):
        """何も考えずにとりあえずこれを実行すれば BigQuery からデータを読み込んで変換し GCS にアップロードしてくれる
        """
        self._logger.info(f"Running with debugging={self.debugging}")
        with tempfile.TemporaryDirectory() as tempdir:
            files: List[str] = []
            train_path = os.path.join(tempdir, f"{self.name}_training.ftr")

            self.read_and_save_features(
                self.train_table, train_path,
            )
            self._upload_to_gs([train_path])

    def read_and_save_features(
        self,
        train_table_name: str,
        train_output_path: str,
    ) -> None:
        df_train_input = self._read_from_bigquery(train_table_name)
        df_train_features = self.make_features(
            df_train_input
        )
        assert (
            df_train_input.shape[0] == df_train_features.shape[0]
        ), "generated train features is not compatible with the table"
        df_train_features.columns = f"{self.name}_" + df_train_features.columns

        if self.save_memory:
            self._logger.info("Reduce memory size - train data")
            df_train_features = reduce_mem_usage(df_train_features)

        self._logger.info(f"Saving features to {train_output_path}")
        df_train_features.to_feather(train_output_path)

    def _read_from_bigquery(self, table_name: str) -> pd.DataFrame:
        self._logger.info(f"Reading from {table_name}")
        query = """
            select {}
            from {}
            where content_type_id = 0
        """.format(
            ", ".join(self.import_columns()), table_name
        )
        if self.debugging:
            query += " limit 10000"

        bqclient = bigquery.Client(project=PROJECT_ID)
        bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient()
        df = (
            bqclient.query(query)
            .result()
            .to_dataframe(bqstorage_client=bqstorageclient)
        )
        return df

    def _upload_to_gs(self, files: List[str]):
        client = storage.Client(project=PROJECT_ID)
        bucket = client.get_bucket(GCS_BUCKET_NAME)

        if self.debugging:
            bucket_dir_name = "features_debug"
        else:
            bucket_dir_name = "features"

        for filename in files:
            basename = os.path.basename(filename)
            blob = storage.Blob(os.path.join(bucket_dir_name, basename), bucket)
            self._logger.info(f"Uploading {basename} to {blob.path}")
            blob.upload_from_filename(filename)

    def _download_from_gs(self, feather_file_name: str) -> pd.DataFrame:
        """GCSにある特徴量ファイル(feather形式)を読み込む
        """
        client = storage.Client(project=PROJECT_ID)
        bucket = client.get_bucket(GCS_BUCKET_NAME)

        if self.debugging:
            bucket_dir_name = "features_debug"
        else:
            bucket_dir_name = "features"

        blob = storage.Blob(
            os.path.join(bucket_dir_name, feather_file_name),
            bucket
        )
        content = blob.download_as_string()
        print(f"Downloading {feather_file_name} from {blob.path}")
        df = pd.read_feather(BytesIO(content))

        return df

