import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class UserCumulativeUserInteractionsByPart(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(self) -> pd.DataFrame:
        query = """
          WITH
          global_mean AS (
            SELECT
              AVG(answered_correctly) AS mean,
            FROM
              `wantedly-individual-shu.riiid.train_questions`
            WHERE
              val = 0   -- use only train
          ),
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
              AVG(answered_correctly) AS accuracy
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
              content_id,
              -- leakを防ぐために, 過去レコードから現時点の1つ前のレコードまでを計算範囲とする
              SUM(1) OVER (PARTITION BY user_id, part ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS cumlative_user_appearance,
              SUM(answered_correctly) OVER (PARTITION BY user_id, part ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS cumlative_user_corrected_answers,
            FROM
              `wantedly-individual-shu.riiid.train_questions`
          ),
          summary AS (
            SELECT
              row_id,
              content_id,
              cumlative_user_appearance,
              cumlative_user_corrected_answers,
              (cumlative_user_corrected_answers + 20 * (SELECT mean FROM global_mean))/ (cumlative_user_appearance + 20) AS mean_user_accuracy,
            FROM
              cumulative
          )
          SELECT
            summary.cumlative_user_appearance,
            summary.cumlative_user_corrected_answers,
            summary.mean_user_accuracy,
            2 * (aggregation_per_content.accuracy *  summary.mean_user_accuracy) / (aggregation_per_content.accuracy + summary.mean_user_accuracy) AS hmean_user_content_accuracy,
          FROM
            summary
          LEFT OUTER JOIN
            aggregation_per_content
            ON summary.content_id = aggregation_per_content.content_id
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
    UserCumulativeUserInteractionsByPart.main()
