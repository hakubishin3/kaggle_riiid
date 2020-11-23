import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class Fold(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(self) -> pd.DataFrame:
        query = """
            WITH
            train_only_questions AS (
              SELECT
                row_id,
              FROM
                riiid.train
              WHERE
                content_type_id = 0
            ),
            fold AS (
            SELECT
              train_only_questions.row_id,
              IF(val_row_id.row_id IS NOT NULL, 1, 0) AS val,
            FROM
              train_only_questions
            LEFT OUTER JOIN
              riiid.val_row_id AS val_row_id
              ON train_only_questions.row_id = val_row_id.row_id
            )
            SELECT
              val,
            FROM
              fold
        """
        query += " order by row_id"
        if self.debugging:
            query += " limit 10000"
        self._logger.info(f"{query}")

        bqclient = bigquery.Client(project=self.PROJECT_ID)
        bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient()
        df = (
            bqclient.query(query)
            .result()
            .to_dataframe(bqstorage_client=bqstorageclient)
        )
        return df

    def make_features(self, df_train_input):
        # read features
        df_train_features = self._read_features_from_bigquery()
        print(df_train_features.shape)
        print(df_train_features.isnull().sum())

        return df_train_features


if __name__ == "__main__":
    Fold.main()
