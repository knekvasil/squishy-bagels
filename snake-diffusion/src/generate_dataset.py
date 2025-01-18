import click
import yaml

from utils.utils import EasyDict, instantiate_from_config
from q_agent import QAgent, QAgentConfig
from game.env import Environment

@click.command()
@click.option('--config', help='Config for generation', metavar='YAML', type=str, required=True, default="config/SnakeAgent.yaml")
@click.option('--model', help='Path to save model', type=str, required=True)
@click.option('--dataset', help='Path to save dataset', type=str, required=True)
@click.option('--record', help='Record actions and snapshots', is_flag=True)
@click.option('--clear-dataset', help='Clear dataset folder', is_flag=True, required=False, default=False)
@click.option('--show-plot', help='Show plot', is_flag=True, required=False, default=False)
@click.option('--last-checkpoint', help='Path of checkpoint to resume the training', type=str, required=False)
def main(**kwargs):
    options = EasyDict(kwargs)
    with open(options.config, 'r') as f:
        config = EasyDict(**yaml.safe_load(f))
    env: Environment = instantiate_from_config(config.env)
    q_agent_config = QAgentConfig(**instantiate_from_config(config.q_agent))
    q_agent = QAgent(env, q_agent_config, options.model, options.dataset, options.get("last_checkpoint", None))
    q_agent.train(options.show_plot, options.record, options.clear_dataset)

if __name__ == "__main__":
    main()