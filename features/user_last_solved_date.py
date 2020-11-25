import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class UserLastSolvedDate(BaseFeature):
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
              content_id,
              answered_correctly,
              Floor(timestamp /1000 / 60 / 60 / 24) AS date
            FROM
              `wantedly-individual-shu.riiid.train_questions`
          ),
          aggregation_by_user_date AS (
            WITH
            agg AS (
              SELECT
                user_id,
                date,
                COUNT(*) AS cnt,
                SUM(answered_correctly) / COUNT(*) AS user_accuracy,
              FROM
                train
              GROUP BY
                user_id,
                date
            )
            SELECT
              user_id,
              date,
              date - LAG(date) OVER(PARTITION BY user_id ORDER BY date) AS elapsed_date,
              LAG(cnt) OVER(PARTITION BY user_id ORDER BY date) AS last_date_answers,
              LAG(user_accuracy) OVER(PARTITION BY user_id ORDER BY date) AS last_date_user_accuracy,
            FROM
              agg
          ),
          aggregation_per_content AS (
            SELECT
              content_id,
              AVG(answered_correctly) AS mean_content_accuracy
            FROM
              `wantedly-individual-shu.riiid.train_questions`
            WHERE
              val = 0   -- use only train
            GROUP BY
              content_id
          )
          SELECT
            IFNULL(aggregation_by_user_date.elapsed_date, -1) AS elapsed_date,
            --IFNULL(aggregation_by_user_date.last_date_answers, -1) AS last_date_answers,
            --IFNULL(aggregation_by_user_date.last_date_user_accuracy, -1) AS last_date_user_accuracy,
            --CASE WHEN aggregation_per_content.mean_content_accuracy = 0 THEN -1
            --      ELSE (aggregation_by_user_date.last_date_user_accuracy - aggregation_per_content.mean_content_accuracy) / aggregation_per_content.mean_content_accuracy END AS change_rate_mean_content_accuracy_and_last_date_user_accuracy,
          FROM
            train
          LEFT OUTER JOIN
            aggregation_by_user_date
            ON train.user_id = aggregation_by_user_date.user_id
            AND train.date = aggregation_by_user_date.date
          LEFT OUTER JOIN
            aggregation_per_content
            ON train.content_id = aggregation_per_content.content_id
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
    UserLastSolvedDate.main()
