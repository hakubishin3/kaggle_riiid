import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class UserLastSolvedProblem(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(self) -> pd.DataFrame:
        query = """
            WITH
            train AS (
              SELECT
                content_id,
                val,
                answered_correctly,
                row_number() over(partition by user_id, content_id order by timestamp) as rank
              FROM
                `wantedly-individual-shu.riiid.train_questions`
            ),
            aggregation_per_content AS (
              SELECT
                content_id,
                AVG(answered_correctly) AS mean_content_accuracy
              FROM
                train
              WHERE
                val = 0   -- use only train
                AND rank = 1   -- 各ユーザの1回目の正解データのみ使う
              GROUP BY
                content_id
            )
            SELECT
              LAG(train_questions.answered_correctly) OVER(PARTITION BY train_questions.user_id ORDER BY train_questions.timestamp) AS prior_answered_correctly,
              IF(train_questions.part = LAG(train_questions.part) OVER(PARTITION BY train_questions.user_id ORDER BY train_questions.timestamp), 1, 0) AS prior_same_part,
              LAG(aggregation_per_content.mean_content_accuracy) OVER(PARTITION BY train_questions.user_id ORDER BY train_questions.timestamp) AS prior_mean_content_accuracy,
              CASE WHEN aggregation_per_content.mean_content_accuracy = 0 THEN -999
                   ELSE (LAG(aggregation_per_content.mean_content_accuracy) OVER(PARTITION BY train_questions.user_id ORDER BY train_questions.timestamp) - aggregation_per_content.mean_content_accuracy) / aggregation_per_content.mean_content_accuracy END AS change_rate_mean_content_accuracy,
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
    UserLastSolvedProblem.main()
