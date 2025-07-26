from dotenv import load_dotenv
load_dotenv()

import hydra
from omegaconf import DictConfig

from gmumcl.experiment import ExperimentPipeline


def print_metrics_callback(pipeline, locals):
    print('R:', locals['R'])

@hydra.main(version_base=None, config_path="pkg://gmumcl/config")
def main(config: DictConfig):
    pipeline = ExperimentPipeline(config, callbacks=[print_metrics_callback])
    metrics = pipeline.run()
    print('Final metrics:', metrics)

if __name__ == "__main__":
    main()