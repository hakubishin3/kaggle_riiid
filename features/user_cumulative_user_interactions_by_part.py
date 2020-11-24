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
            aggregation_by_content AS (
              SELECT
                content_id,
                AVG(answered_correctly) AS accuracy
              FROM
                `wantedly-individual-shu.riiid.train_questions`
              WHERE
                val = 0
              GROUP BY
                content_id
            ),
            aggregation_by_part AS (
              SELECT
                part,
                AVG(answered_correctly) AS accuracy
              FROM
                `wantedly-individual-shu.riiid.train_questions`
              WHERE
                val = 0
              GROUP BY
                part
            ),
            cumulative AS (
              SELECT
                row_id,
                content_id,
                part,
                -- leakを防ぐために, 過去レコードから現時点の1つ前のレコードまでを計算範囲とする
              SUM(1) OVER (PARTITION BY user_id, part ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS cumlative_user_appearance,
              SUM(answered_correctly) OVER (PARTITION BY user_id, part ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS cumlative_user_corrected_answers,
              FROM
                `wantedly-individual-shu.riiid.train_questions`
            ),
            summary AS (
              SELECT
                cumulative.row_id,
                cumlative_user_appearance,
                cumlative_user_corrected_answers,
                cumlative_user_corrected_answers / cumlative_user_appearance AS mean_user_accuracy,
                aggregation_by_content.accuracy AS mean_content_accuracy,
                aggregation_by_part.accuracy AS mean_part_accuracy,
              FROM
                cumulative
              LEFT OUTER JOIN
                aggregation_by_content
                ON cumulative.content_id = aggregation_by_content.content_id
              LEFT OUTER JOIN
                aggregation_by_part
                ON cumulative.part = aggregation_by_part.part
            )
            SELECT
              cumlative_user_appearance,
              cumlative_user_corrected_answers,
              mean_user_accuracy,
              2 * (mean_user_accuracy *  mean_content_accuracy) / (mean_user_accuracy + mean_content_accuracy) AS hmean_user_content_accuracy,
              2 * (mean_user_accuracy *  mean_part_accuracy) / (mean_user_accuracy + mean_part_accuracy) AS hmean_user_part_accuracy,
              3 * (mean_user_accuracy *  mean_content_accuracy * mean_part_accuracy) / (mean_user_accuracy * mean_content_accuracy + mean_user_accuracy * mean_part_accuracy + mean_part_accuracy * mean_content_accuracy) AS hmean_user_part_content_accuracy,
            FROM
              summary
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
