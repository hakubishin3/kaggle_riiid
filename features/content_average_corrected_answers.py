import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class ContentAverageCorrectedAnswers(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(self) -> pd.DataFrame:
        query = """
          WITH
          aggregation_per_content AS (
            SELECT
              content_id,
              AVG(answered_correctly) AS accuracy
            FROM
              `wantedly-individual-shu.riiid.train_questions`
            WHERE
              val = 0   -- use only train
            GROUP BY
              content_id
          )
          SELECT
            aggregation_per_content.accuracy AS mean_content_accuracy,
          FROM
            `wantedly-individual-shu.riiid.train_questions` AS train_questions
          LEFT OUTER JOIN
            aggregation_per_content
            ON train_questions.content_id = aggregation_per_content.content_id
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
    ContentAverageCorrectedAnswers.main()
