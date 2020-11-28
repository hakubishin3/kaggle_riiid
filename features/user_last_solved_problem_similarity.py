import pandas as pd
from base import BaseFeature
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1


class UserLastSolvedProblemSimilarity(BaseFeature):
    def import_columns(self):
        return [
            "1",
        ]

    def _read_features_from_bigquery(self) -> pd.DataFrame:
        query = """
          WITH
          train AS (
            SELECT
              row_id,
              user_id,
              content_id,
              LAG(content_id, 1) OVER(PARTITION BY user_id ORDER BY timestamp) AS prior_content_id,
            FROM
              `wantedly-individual-shu.riiid.train_questions` AS train_questions
          )
          SELECT
            CASE WHEN
              SQRT(POW(user_vec.NMF_0, 2) + POW(user_vec.NMF_1, 2) + POW(user_vec.NMF_2, 2) + POW(user_vec.NMF_3, 2) + POW(user_vec.NMF_4, 2) + POW(user_vec.NMF_5, 2) + POW(user_vec.NMF_6, 2) + POW(user_vec.NMF_7, 2) + POW(user_vec.NMF_8, 2) + POW(user_vec.NMF_9, 2)) = 0
              OR
              SQRT(POW(content_vec.NMF_0, 2) + POW(content_vec.NMF_1, 2) + POW(content_vec.NMF_2, 2) + POW(content_vec.NMF_3, 2) + POW(content_vec.NMF_4, 2) + POW(content_vec.NMF_5, 2) + POW(content_vec.NMF_6, 2) + POW(content_vec.NMF_7, 2) + POW(content_vec.NMF_8, 2) + POW(content_vec.NMF_9, 2)) = 0
              THEN -999
            ELSE
              (
                user_vec.NMF_0 * content_vec.NMF_0 + 
                user_vec.NMF_1 * content_vec.NMF_1 + 
                user_vec.NMF_2 * content_vec.NMF_2 + 
                user_vec.NMF_3 * content_vec.NMF_3 + 
                user_vec.NMF_4 * content_vec.NMF_4 + 
                user_vec.NMF_5 * content_vec.NMF_5 + 
                user_vec.NMF_6 * content_vec.NMF_6 + 
                user_vec.NMF_7 * content_vec.NMF_7 + 
                user_vec.NMF_8 * content_vec.NMF_8 + 
                user_vec.NMF_9 * content_vec.NMF_9
              ) / SQRT(POW(user_vec.NMF_0, 2) + POW(user_vec.NMF_1, 2) + POW(user_vec.NMF_2, 2) + POW(user_vec.NMF_3, 2) + POW(user_vec.NMF_4, 2) + POW(user_vec.NMF_5, 2) + POW(user_vec.NMF_6, 2) + POW(user_vec.NMF_7, 2) + POW(user_vec.NMF_8, 2) + POW(user_vec.NMF_9, 2))
                / SQRT(POW(content_vec.NMF_0, 2) + POW(content_vec.NMF_1, 2) + POW(content_vec.NMF_2, 2) + POW(content_vec.NMF_3, 2) + POW(content_vec.NMF_4, 2) + POW(content_vec.NMF_5, 2) + POW(content_vec.NMF_6, 2) + POW(content_vec.NMF_7, 2) + POW(content_vec.NMF_8, 2) + POW(content_vec.NMF_9, 2))
            END AS similarity,
          FROM
            train AS train
          LEFT OUTER JOIN
            `wantedly-individual-shu.riiid.content_embedding_nmf10_v2` AS content_vec
            ON train.content_id = content_vec.content_id
          LEFT OUTER JOIN
            `wantedly-individual-shu.riiid.content_embedding_nmf10_v2` AS user_vec
            ON train.prior_content_id = user_vec.content_id
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
    UserLastSolvedProblemSimilarity.main()
