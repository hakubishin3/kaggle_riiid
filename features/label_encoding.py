import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class LabelEncoding(BaseFeature):
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
                content_id,
                prior_question_had_explanation,
              FROM
                riiid.train
              WHERE
                content_type_id = 0
            )
            SELECT
              questions.part,
              CASE WHEN train_only_questions.prior_question_had_explanation IS TRUE THEN 1 ELSE 0 END AS prior_question_had_explanation,
            FROM
              train_only_questions
            LEFT OUTER JOIN
              riiid.questions
              ON train_only_questions.content_id = questions.question_id
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
    LabelEncoding.main()
