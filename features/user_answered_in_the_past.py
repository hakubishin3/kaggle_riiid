import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class UserAnsweredInThePast(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(self) -> pd.DataFrame:
        query = """
          WITH
          cumulative AS (
            SELECT
              row_id,
              user_id,
              timestamp,
              content_id,
              SUM(1) OVER (PARTITION BY user_id, content_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) - 1 AS number_of_answered_the_sampe_question,
              SUM(1) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS cumlative_user_appearance,
              SUM(1) OVER (PARTITION BY user_id, content_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS cumlative_user_appearance_per_content,
              SUM(answered_correctly) OVER (PARTITION BY user_id, content_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS cumlative_user_corrected_answers_per_content,
              LAG(timestamp, 1) OVER(PARTITION BY user_id, content_id ORDER BY timestamp) AS lag_1_timestamp,
              LAG(answered_correctly, 1) OVER(PARTITION BY user_id, content_id ORDER BY timestamp) AS lag_1_answered_correctly,
              LAG(answered_correctly, 2) OVER(PARTITION BY user_id, content_id ORDER BY timestamp) AS lag_2_answered_correctly,
              LAG(answered_correctly, 3) OVER(PARTITION BY user_id, content_id ORDER BY timestamp) AS lag_3_answered_correctly,
              LAG(answered_correctly, 4) OVER(PARTITION BY user_id, content_id ORDER BY timestamp) AS lag_4_answered_correctly,
              LAG(answered_correctly, 5) OVER(PARTITION BY user_id, content_id ORDER BY timestamp) AS lag_5_answered_correctly,
              LAG(answered_correctly, 6) OVER(PARTITION BY user_id, content_id ORDER BY timestamp) AS lag_6_answered_correctly,
              LAG(answered_correctly, 7) OVER(PARTITION BY user_id, content_id ORDER BY timestamp) AS lag_7_answered_correctly,
              LAG(answered_correctly, 8) OVER(PARTITION BY user_id, content_id ORDER BY timestamp) AS lag_8_answered_correctly,
              LAG(answered_correctly, 9) OVER(PARTITION BY user_id, content_id ORDER BY timestamp) AS lag_9_answered_correctly,
              LAG(answered_correctly, 10) OVER(PARTITION BY user_id, content_id ORDER BY timestamp) AS lag_10_answered_correctly,
            FROM
              `wantedly-individual-shu.riiid.train_questions`
          )
          SELECT
            -- 過去に同じ問題を何回解いたか
            number_of_answered_the_sampe_question,
            -- 過去の同じ問題の正解率
            cumlative_user_corrected_answers_per_content / cumlative_user_appearance_per_content AS mean_user_accuracy,
            -- 過去に同じ問題を解いてから何時間経過しているか
            timestamp - lag_1_timestamp AS diff_timestamp,
            cumlative_user_appearance - LAG(cumlative_user_appearance, 1) OVER (PARTITION BY user_id, content_id ORDER BY timestamp) - 1 AS diff_answers,
            -- 過去に同じ問題を解いてから今に至るまでの平均時間回答数
            CASE WHEN cumlative_user_appearance - LAG(cumlative_user_appearance, 1) OVER (PARTITION BY user_id, content_id ORDER BY timestamp) - 1 = 0 THEN -999
              ELSE (timestamp - lag_1_timestamp) / (cumlative_user_appearance - LAG(cumlative_user_appearance, 1) OVER (PARTITION BY user_id, content_id ORDER BY timestamp) - 1) END AS differential,
            -- 過去の同じ問題に対する正解・不正解を時系列順に並べてfloatに(seq2vecと同じやり方)
            IFNULL(lag_1_answered_correctly, 0) +
              0.1 * IFNULL(lag_2_answered_correctly, 0) +
              0.01 * IFNULL(lag_3_answered_correctly, 0) +
              0.001 * IFNULL(lag_4_answered_correctly, 0) +
              0.0001 * IFNULL(lag_5_answered_correctly, 0) +
              0.00001 * IFNULL(lag_6_answered_correctly, 0) +
              0.000001 * IFNULL(lag_7_answered_correctly, 0) +
              0.0000001 * IFNULL(lag_8_answered_correctly, 0) + 
              0.00000001 * IFNULL(lag_9_answered_correctly, 0) +
              0.000000001 * IFNULL(lag_10_answered_correctly, 0) AS seq2vec
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
    UserAnsweredInThePast.main()
