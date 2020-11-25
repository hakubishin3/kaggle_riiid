import json
import pathlib
import argparse
import numpy as np
import pandas as pd
from src.utils import seed_everything, get_logger, json_dump, upload_to_gcs
from src.feature_loader import FeatureLoader
from src.runner import Runner
from src.models.lightgbm import ModelLightGBM
from multiprocessing import cpu_count


seed_everything(71)

model_map = {
    'lightgbm': ModelLightGBM,
}


def main():
    # =========================================
    # === Settings
    # =========================================
    # Get logger
    logger = get_logger(__name__)
    logger.info('Settings')

    # Get argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='model_lgb_hakubishin_20200317/configs/model_0.json')
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logger.info(f'config: {args.config}')
    logger.info(f'debug: {args.debug}')

    # Get config
    config = json.load(open(args.config))
    config.update({
        'args': {
            'config': args.config,
            'debug': args.debug
        }
    })
    config["model"]["model_params"]["nthread"] = cpu_count()

    # Create a directory for model output
    model_no = pathlib.Path(args.config).stem
    model_output_dir = (
        pathlib.Path(config['model_dir_name']) /
        pathlib.Path(config['dataset']['output_directory']) / model_no
    )
    if not model_output_dir.exists():
        model_output_dir.mkdir()

    logger.info(f'model_output_dir: {str(model_output_dir)}')
    logger.debug(f'model_output_dir exists: {model_output_dir.exists()}')
    config.update({
        'model_output_dir': str(model_output_dir)
    })

    # =========================================
    # === Loading features
    # =========================================
    logger.info('Loading features')
    logger.info(f'targets: {config["target"]}')
    logger.info(f'features: {config["features"]}')

    # features
    x_train = FeatureLoader(
        data_type="training", debugging=args.debug
        ).load_features(config["features"])

    # targets
    y_train_set = FeatureLoader(
        data_type="training", debugging=args.debug
        ).load_features(config["target"])

    # folds
    folds_train = FeatureLoader(
        data_type="training", debugging=args.debug
        ).load_features(config["folds"])

    logger.debug(f'y_train_set: {y_train_set.shape}')
    logger.debug(f'x_train: {x_train.shape}')

    # =========================================
    # === Train model and predict
    # =========================================
    logger.info('Train model and predict')

    # Get target values
    y_train = y_train_set["Target_answered_correctly"].values

    # Get folds
    trn_idx = folds_train.query("Fold_val != 1").index
    val_idx = folds_train.query("Fold_val == 1").index
    folds_ids = [(trn_idx, val_idx)]
    logger.debug(f"n_trn={len(trn_idx)}, n_val={len(val_idx)}")
    logger.debug(f"trn_pos={y_train[trn_idx].sum()}, val_pos={y_train[val_idx].sum()}")

    # Train and predict
    model_cls = model_map[config['model']['name']]
    model_params = config['model']
    runner = Runner(
        model_cls, model_params, model_output_dir, f'{model_cls.__name__}', n_fold=1,
    )
    oof_preds, evals_result, importances = runner.train_cv(
        x_train, y_train, folds_ids)
    config.update(evals_result)

    # Save importances
    importances.mean(axis=1).reset_index().rename(
        columns={"index": "feature", 0: "value"}
    ).sort_values("value", ascending=False).to_csv(
        model_output_dir / "importances.csv", index=False
    )

    # Save oof-pred file
    oof_preds_file_name = f"oof_pred"
    np.save(model_output_dir / oof_preds_file_name, oof_preds)
    logger.info(f'Save oof-pred file: {model_output_dir/ oof_preds_file_name}')

    # Save files (override)
    logger.info('Save files')
    save_path = model_output_dir / 'output.json'
    json_dump(config, save_path)
    logger.info(f'Save model log: {save_path}')

    # =========================================
    # === Upload to GCS
    # =========================================
    if not args.debug:
        logger.info('Upload to GCS')

        bucket_dir_name = config["model_dir_name"] + "/" + model_no
        logger.info(f'bucket_dir_name: {bucket_dir_name}')

        files = list(model_output_dir.iterdir())
        upload_to_gcs(bucket_dir_name, files)


if __name__ == '__main__':
    main()

