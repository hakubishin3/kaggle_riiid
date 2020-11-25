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
            SELECT
              LAG(answered_correctly) OVER(PARTITION BY user_id ORDER BY timestamp) AS prior_answered_correctly,
              IF(part = LAG(part) OVER(PARTITION BY user_id ORDER BY timestamp), 1, 0) AS prior_same_part,
            FROM
            `wantedly-individual-shu.riiid.train_questions`
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
