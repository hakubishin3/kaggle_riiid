import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class UserCumulativeQuestionElapsedTime(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(self) -> pd.DataFrame:
        query = """
            WITH
            train AS (
              SELECT
                *,
                LAG(answered_correctly) OVER(PARTITION BY user_id ORDER BY timestamp) AS prior_answered_correctly,
              FROM
                `wantedly-individual-shu.riiid.train_questions`
            ),
            cumulative AS (
              SELECT
                row_id,
                -- leakを防ぐために, 過去レコードから現時点の1つ前のレコードまでを計算範囲とする
                SUM(1) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS cumlative_user_appearance,
                SUM(answered_correctly) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS cumlative_user_corrected_answers,
                SUM(prior_question_elapsed_time) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumlative_question_elapsed_time,
                SUM(prior_question_elapsed_time * prior_answered_correctly) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumlative_answered_question_elapsed_time,
                SUM(prior_question_elapsed_time * (1 - prior_answered_correctly)) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumlative_not_answered_question_elapsed_time,
              FROM
                train
            )
            SELECT
              CAST(cumlative_question_elapsed_time AS INT64) AS cumlative_question_elapsed_time,
              CAST(cumlative_answered_question_elapsed_time AS INT64) AS cumlative_answered_question_elapsed_time,
              CAST(cumlative_not_answered_question_elapsed_time AS INT64) AS cumlative_not_answered_question_elapsed_time,
              cumlative_question_elapsed_time / cumlative_user_appearance AS mean_question_elapsed_time,
              CASE
                WHEN cumlative_user_corrected_answers = 0 THEN 0
                ELSE cumlative_answered_question_elapsed_time / cumlative_user_corrected_answers END AS mean_answered_question_elapsed_time,
              CASE
                WHEN cumlative_user_appearance - cumlative_user_corrected_answers = 0 THEN 0
                ELSE cumlative_not_answered_question_elapsed_time / (cumlative_user_appearance - cumlative_user_corrected_answers) END AS mean_not_answered_question_elapsed_time,
              CASE
                WHEN cumlative_user_corrected_answers = 0 OR cumlative_user_appearance - cumlative_user_corrected_answers = 0 THEN NULL
                ELSE cumlative_answered_question_elapsed_time / cumlative_user_corrected_answers - cumlative_not_answered_question_elapsed_time / (cumlative_user_appearance - cumlative_user_corrected_answers) END AS diff_answered_not_answered_question_elapsed_time,
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
        return df

    def make_features(self, df_train_input):
        # read features
        df_train_features = self._read_features_from_bigquery()
        print(df_train_features.shape)
        print(df_train_features.isnull().sum())

        return df_train_features


if __name__ == "__main__":
    UserCumulativeQuestionElapsedTime.main()
