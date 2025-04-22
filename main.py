import sys
from src.train import main as train_main
from src.utils import register_env

register_env()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train_main(config_path="configs/agent.yaml")
    else:
        print("Unknown command. Use 'train'.")