import argparse
import configparser
import os
import time

from xgb_train import load_data, train_model

start_time = time.time()


def load_config(config_path):
    config_parse = configparser.ConfigParser()
    config_parse.read(config_path)
    return config_parse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini',
                        help='Path to configuration file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        exit(1)

    config = load_config(args.config)
    X_train, y_train, X_val, y_val = load_data(config)
    train_model(config, X_train, y_train, X_val, y_val)

    print(f"use time: {time.time() - start_time}")
