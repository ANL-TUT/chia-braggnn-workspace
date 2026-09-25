"""Pass config_alphaevolve.yaml's ``alphaevolve.evolution_settings`` (pareto
sampling) through to the AlphaEvolve experiment.

skydiscover's AlphaEvolve backend (the fork baked into the evolver
image) builds the experiment config from a fixed set of keys and drops
``evolution_settings``. The evolver image cannot be rebuilt mid-run, so this
wraps the builder instead. It takes effect in the EvolverNode actor's process
because braggnn_evaluator imports it, and the evaluator is unpickled there
before the search builds the experiment. Remove it once the backend passes the
key itself.
"""

import logging

from skydiscover.extras.external import alphaevolve_backend as _backend

logger = logging.getLogger(__name__)

_PASSED_THROUGH = ("evolution_settings",)


def _patch() -> None:
    original = _backend._build_experiment_config
    if getattr(original, "_chia_passes_evolution_settings", False):
        return

    def build_experiment_config(config_obj, ae_config, iterations):
        exp_config = original(config_obj, ae_config, iterations)
        for key in _PASSED_THROUGH:
            if key in ae_config and key not in exp_config:
                exp_config[key] = ae_config[key]
                logger.info("AlphaEvolve experiment %s: %s", key, ae_config[key])
        return exp_config

    build_experiment_config._chia_passes_evolution_settings = True
    _backend._build_experiment_config = build_experiment_config


_patch()
