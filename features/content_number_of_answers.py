import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class ContentNumberOfAnswers(BaseFeature):
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
              user_id,
              content_id,
            FROM
              riiid.train
            WHERE
              content_type_id = 0
          ),
          aggregation_per_content AS (
            SELECT
              content_id,
              COUNT(*) AS answers,
              COUNT(DISTINCT user_id) AS answered_users,
              COUNT(*) / COUNT(DISTINCT user_id) AS answers_per_user,
            FROM
              train_only_questions
            GROUP BY
              content_id
          )
          SELECT
            aggregation_per_content.answers,
            aggregation_per_content.answered_users,
            aggregation_per_content.answers_per_user,
          FROM
            train_only_questions
          LEFT OUTER JOIN
            aggregation_per_content
            ON train_only_questions.content_id = aggregation_per_content.content_id
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
    ContentNumberOfAnswers.main()
