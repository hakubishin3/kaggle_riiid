import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class UserCumulativeUserInteractionsWeightedWindowSize100(BaseFeature):
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
                CASE
                  WHEN AVG(answered_correctly) = 0 THEN 999
                  ELSE (1 - AVG(answered_correctly)) / AVG(answered_correctly) END AS weight
                FROM
                train
                WHERE
                val = 0   -- use only train
                AND rank = 1   -- 各ユーザの1回目の正解データのみ使う
                GROUP BY
                content_id
            ),
            cumulative AS (
              SELECT
                row_id,
                -- leakを防ぐために, 過去レコードから現時点の1つ前のレコードまでを計算範囲とする
                SUM(aggregation_per_content.weight) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING) AS cumlative_user_appearance,
                SUM(answered_correctly * aggregation_per_content.weight) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING) AS cumlative_user_corrected_answers,
              FROM
                `wantedly-individual-shu.riiid.train_questions` AS train_questions
              LEFT OUTER JOIN
                  aggregation_per_content
                  ON train_questions.content_id = aggregation_per_content.content_id
            )
            SELECT
              cumlative_user_appearance,
              cumlative_user_corrected_answers,
              cumlative_user_corrected_answers / cumlative_user_appearance AS mean_user_accuracy,
            FROM
              cumulative
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
    UserCumulativeUserInteractionsWeightedWindowSize100.main()
