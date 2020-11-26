import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class UserCumulativeUserInteractionsWindowFreq(BaseFeature):
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
                -- leakを防ぐために, 過去レコードから現時点の1つ前のレコードまでを計算範囲とする
                -- by user_id
                -- 5 ~
                SUM(1) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS cumlative_user_appearance_5,
                SUM(answered_correctly) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS cumlative_user_corrected_answers_5,
                -- 10 ~
                SUM(1) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS cumlative_user_appearance_10,
                SUM(answered_correctly) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS cumlative_user_corrected_answers_10,
                -- by user_id and part
                -- 5 ~
                SUM(1) OVER (PARTITION BY user_id, part ORDER BY timestamp ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS cumlative_user_appearance_5_by_part,
                SUM(answered_correctly) OVER (PARTITION BY user_id, part ORDER BY timestamp ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS cumlative_user_corrected_answers_5_by_part,
                -- 10 ~
                SUM(1) OVER (PARTITION BY user_id, part ORDER BY timestamp ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS cumlative_user_appearance_10_by_part,
                SUM(answered_correctly) OVER (PARTITION BY user_id, part ORDER BY timestamp ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS cumlative_user_corrected_answers_10_by_part,
              FROM
                `wantedly-individual-shu.riiid.train_questions`
            )
            SELECT
              cumlative_user_corrected_answers_5 / cumlative_user_appearance_5 AS mean_user_accuracy_5,
              cumlative_user_corrected_answers_10 / cumlative_user_appearance_10 AS mean_user_accuracy_10,
              cumlative_user_corrected_answers_5_by_part / cumlative_user_appearance_5_by_part AS mean_user_accuracy_5_by_part,
              cumlative_user_corrected_answers_10_by_part / cumlative_user_appearance_10_by_part AS mean_user_accuracy_10_by_part,
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
    UserCumulativeUserInteractionsWindowFreq.main()
