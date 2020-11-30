import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class ContentAverageCorrectedAnswersPosterior(BaseFeature):
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
              content_id,
              lag(content_id, 1) over(partition by user_id order by timestamp)  as bef_content_id,
              part,
              lag(part, 1) over(partition by user_id order by timestamp)  as bef_part,
              answered_correctly,
              lag(answered_correctly, 1) over(partition by user_id order by timestamp)  as bef_answered_correctly,
              val,
              lag(val, 1) over(partition by user_id order by timestamp)  as bef_val,
            FROM
              `wantedly-individual-shu.riiid.train_questions`
          ),
          aggregate as (
            SELECT
              bef_answered_correctly,
              bef_content_id,
              content_id,
              avg(answered_correctly) as mean,
              count(*) as size,
            FROM
              train
            WHERE
              val = 0
              AND bef_val = 0
              AND part = bef_part
            GROUP BY
              bef_answered_correctly,
              bef_content_id,
              content_id
            HAVING
              count(*) >= 100
          )
          SELECT
            aggregate.mean AS mean_content_accuracy,
            aggregate.size AS size,
          FROM
            train
          LEFT OUTER JOIN
            aggregate
            ON train.content_id = aggregate.content_id
            AND train.bef_content_id = aggregate.bef_content_id
            AND train.bef_answered_correctly = aggregate.bef_answered_correctly
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
    ContentAverageCorrectedAnswersPosterior.main()
