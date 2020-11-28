import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class UserCumulativeUserInteractionsByPartWindowSize100(BaseFeature):
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
                SUM(1) OVER (PARTITION BY user_id, part ORDER BY timestamp ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING) AS cumlative_user_appearance,
                SUM(answered_correctly) OVER (PARTITION BY user_id, part ORDER BY timestamp ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING) AS cumlative_user_corrected_answers,
              FROM
                `wantedly-individual-shu.riiid.train_questions`
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
    UserCumulativeUserInteractionsByPartWindowSize100.main()
