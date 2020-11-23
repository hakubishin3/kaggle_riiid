import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class CumulativeAppearancesPerUser(BaseFeature):
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
                timestamp,
                answered_correctly,
              FROM
                riiid.train
              WHERE
                content_type_id = 0
            ),
            cumulative AS (
              SELECT
                row_id,
                SUM(1) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumlative_appearance,
                SUM(answered_correctly) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumlative_corrected_answers,
              FROM
                train_only_questions
            )
            SELECT
              cumlative_appearance,
              cumlative_corrected_answers,
              cumlative_corrected_answers / cumlative_appearance AS average_corrected_answers,
            FROM
              cumulative
        """
        self._logger.info(f"{query}")
        query += " order by row_id"
        if self.debugging:
            query += " limit 10000"

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
    CumulativeAppearancesPerUser.main()
