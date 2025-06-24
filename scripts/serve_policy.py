from test.test_gzip import data1
from test.test_shlex import data
import dataclasses
import enum
import logging
import socket
import openpi.transforms as transforms

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""

@dataclasses.dataclass
class Fixed:
    """Use a fixed policy for the given environment."""

@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default | Fixed = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="gs://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="gs://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="gs://openpi-assets/checkpoints/pi0_fast_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Fixed():
            fixed_norm_stats: dict[str, transforms.NormStats] = {
                "norm_stats": { "actions": {
                  "mean": [
                    -0.00016014573662360493,
                    -0.0035431239667740957,
                    0.00047960496805258873,
                    0.0014334739524824414,
                    0.0001206908110824996,
                    0.0011677667606468747,
                    0.0005674554334817187,
                    0.4729264108026316,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                  ],
                  "std": [
                    0.14039742770161961,
                    0.17483806634652702,
                    0.11945758846211159,
                    0.18796604550815885,
                    0.2399466744249755,
                    0.24966444084962108,
                    0.2946190219814127,
                    0.4120095862153511,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                  ],
                  "q01": [
                    -0.466942831301689,
                    -0.5403164483971585,
                    -0.4097102898055547,
                    -0.5929721020698546,
                    -0.8564867628097532,
                    -0.8256900206917261,
                    -1.0080286278496486,
                    0.004200000000000001,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                  ],
                  "q99": [
                    0.5355668843984605,
                    0.4520759525710969,
                    0.3826163315719908,
                    0.58378665394783,
                    0.859333964061737,
                    0.73115307846414,
                    1.0392417478020293,
                    0.9998,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                  ]
                },
                "state": {
                  "mean": [
                    -0.6667929585706829,
                    0.5109309789234092,
                    -0.0026893946408467497,
                    -1.9632786655382104,
                    -0.019793539327913272,
                    2.385514920687147,
                    -0.004014034282001113,
                    0.451779177702854,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                  ],
                  "std": [
                    0.46611794449931604,
                    0.2614621230933891,
                    0.3301766343210112,
                    0.34571105674084845,
                    0.5972885780312494,
                    0.4780000295310637,
                    0.8626035486642363,
                    0.3931528670588693,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                  ],
                  "q01": [
                    -1.626023443222046,
                    -0.08749453892707826,
                    -0.8126521904945374,
                    -2.656849225505706,
                    -1.5722319303245222,
                    1.40174001984496,
                    -2.012645048009374,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                  ],
                  "q99": [
                    0.3013750011920928,
                    1.0882165689468382,
                    0.9674761711597442,
                    -1.118987345823913,
                    1.6283248943702429,
                    3.4482878058956277,
                    1.9580948521933959,
                    0.991,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                  ]
                }
              }}

            return _policy_config.create_trained_policy(
                _config.get_config("pi0_fast_droid"), "gs://openpi-assets/checkpoints/pi0_fast_droid", default_prompt=None, norm_stats=fixed_norm_stats
            )

        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
