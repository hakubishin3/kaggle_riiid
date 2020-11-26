import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class UserLecturedAtLeastOnce(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(self) -> pd.DataFrame:
        query = """
          WITH
          train AS (
            SELECT
              row_id,
              user_id,
              timestamp,
              content_type_id,
              IF(content_type_id = 1, 1, 0) AS is_lecture,
              IF(lecture_part = 1, 1, 0) AS is_lecture_part_1,
              IF(lecture_part = 2, 1, 0) AS is_lecture_part_2,
              IF(lecture_part = 3, 1, 0) AS is_lecture_part_3,
              IF(lecture_part = 4, 1, 0) AS is_lecture_part_4,
              IF(lecture_part = 5, 1, 0) AS is_lecture_part_5,
              IF(lecture_part = 6, 1, 0) AS is_lecture_part_6,
              IF(lecture_part = 7, 1, 0) AS is_lecture_part_7,
              IF(lecture_type_of = "concept", 1, 0) AS is_lecture_type_of_concept,
              IF(lecture_type_of = "solving question", 1, 0) AS is_lecture_type_of_solving_question,
              IF(lecture_type_of = "intention", 1, 0) AS is_lecture_type_of_intention,
              IF(lecture_type_of = "starter", 1, 0) AS is_lecture_type_of_starter,
            FROM
              `wantedly-individual-shu.riiid.train_full`
          ),
          add_flg AS (
          SELECT
            row_id,
            content_type_id,
            MAX(is_lecture) OVER(PARTITION BY user_id ORDER BY timestamp RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS is_lecture,
            MAX(is_lecture_part_1) OVER(PARTITION BY user_id ORDER BY timestamp RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS is_lecture_part_1,
            MAX(is_lecture_part_2) OVER(PARTITION BY user_id ORDER BY timestamp RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS is_lecture_part_2,
            MAX(is_lecture_part_3) OVER(PARTITION BY user_id ORDER BY timestamp RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS is_lecture_part_3,
            MAX(is_lecture_part_4) OVER(PARTITION BY user_id ORDER BY timestamp RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS is_lecture_part_4,
            MAX(is_lecture_part_5) OVER(PARTITION BY user_id ORDER BY timestamp RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS is_lecture_part_5,
            MAX(is_lecture_part_6) OVER(PARTITION BY user_id ORDER BY timestamp RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS is_lecture_part_6,
            MAX(is_lecture_part_7) OVER(PARTITION BY user_id ORDER BY timestamp RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS is_lecture_part_7,
            MAX(is_lecture_type_of_concept) OVER(PARTITION BY user_id ORDER BY timestamp RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS is_lecture_type_of_concept,
            MAX(is_lecture_type_of_solving_question) OVER(PARTITION BY user_id ORDER BY timestamp RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS is_lecture_type_of_solving_question,
            MAX(is_lecture_type_of_intention) OVER(PARTITION BY user_id ORDER BY timestamp RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS is_lecture_type_of_intention,
            MAX(is_lecture_type_of_starter) OVER(PARTITION BY user_id ORDER BY timestamp RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS is_lecture_type_of_starter,
          FROM
            train
          )
          SELECT
            is_lecture,
            is_lecture_part_1,
            is_lecture_part_2,
            is_lecture_part_3,
            is_lecture_part_4,
            is_lecture_part_5,
            is_lecture_part_6,
            is_lecture_part_7,
            is_lecture_type_of_concept,
            is_lecture_type_of_solving_question,
            is_lecture_type_of_intention,
            is_lecture_type_of_starter,
          FROM
            add_flg
          WHERE
            content_type_id = 0
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
        print(df)
        return df

    def make_features(self, df_train_input):
        # read features
        df_train_features = self._read_features_from_bigquery()
        print(df_train_features.shape)
        print(df_train_features.isnull().sum())

        return df_train_features


if __name__ == "__main__":
    UserLecturedAtLeastOnce.main()
