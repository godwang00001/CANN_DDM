from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EdgePopConfig:
    legacy_num_E: int | None = None
    tau_E: float = 1
    alpha_E: float = 1
    gamma_E: float = 1
    noise_scale_edge: float = 0
    clamp_frac_E: float = 0.15
    edge_type: str = 'Laplace'
    c_EB: float = 1.
    offset: float = 0
    eb_kernel_mode: str = 'simple'
    eb_kernel_sigma: float = 1.0
    eb_kernel_shift: float = 1.0
    eb_kernel_gain: float = 100.


@dataclass(frozen=True)
class BumpPopConfig:
    legacy_num_B: int | None = None
    tau_B: float = 1
    c_BE: float = 1
    c_BE_params: dict = field(default_factory=lambda: {'mode': 'const'})
    noise_scale_bump: float = 0
    sigma_B: float = 0.25
    sigma_I_BE: float | None = None
    beta_B: float = 1.8
    clamp_frac_B: float = 0.15
    kernel_mode: str = 'gaussian_cann'
    kernel_gain: float = 2.0
    kernel_sigma: float | None = None
    kernel_normed: bool = True
    be_kernel_mode: str = 'simple'
    be_kernel_sigma: float = 1.0
    be_kernel_gain: float = 1.0


@dataclass(frozen=True)
class DecisionSpaceConfig:
    t_start: int = 200
    dur1: int = 100
    dur2: int = 1000
    boundary: float = 1
    drift_rate: float = 0.5
    noise_scale: float = 0.5
    dt_DDM: float = 25
    x0: float = 0.5
    seed: int | None = None


@dataclass(frozen=True)
class NeuralGeometryConfig:
    num_units: int
    coding_limit: float = np.pi / 2
    coding_frac: float | None = 0.3
    clamp_frac: float = 0.15


@dataclass(frozen=True)
class DerivedGeometry:
    num_units: int
    coding_limit: float
    coding_frac: float
    clamp_frac: float
    theta_min: float
    theta_max: float
    coding_theta_min: float
    coding_theta_max: float
    clamp_side_width: int
    left_gap_width: int
    right_gap_width: int
    coding_width: int
    k1: int
    k2: int


def parse_edge_config(edge_params):
    return EdgePopConfig(
        legacy_num_E=edge_params.get('num_E'),
        tau_E=edge_params.get('tau_E', EdgePopConfig.tau_E),
        alpha_E=edge_params.get('alpha_E', EdgePopConfig.alpha_E),
        gamma_E=edge_params.get('gamma_E', EdgePopConfig.gamma_E),
        noise_scale_edge=edge_params.get('noise_scale_edge', EdgePopConfig.noise_scale_edge),
        clamp_frac_E=edge_params.get('clamp_frac_E', EdgePopConfig.clamp_frac_E),
        edge_type=edge_params.get('edge_type', EdgePopConfig.edge_type),
        c_EB=edge_params.get('c_EB', EdgePopConfig.c_EB),
        offset=edge_params.get('offset', EdgePopConfig.offset),
        eb_kernel_mode=edge_params.get('eb_kernel_mode', EdgePopConfig.eb_kernel_mode),
        eb_kernel_sigma=edge_params.get('eb_kernel_sigma', EdgePopConfig.eb_kernel_sigma),
        eb_kernel_shift=edge_params.get('eb_kernel_shift', EdgePopConfig.eb_kernel_shift),
        eb_kernel_gain=edge_params.get('eb_kernel_gain', EdgePopConfig.eb_kernel_gain),
    )


def parse_bump_config(bump_params):
    return BumpPopConfig(
        legacy_num_B=bump_params.get('num_B'),
        tau_B=bump_params.get('tau_B', BumpPopConfig.tau_B),
        c_BE=bump_params.get('c_BE', BumpPopConfig.c_BE),
        c_BE_params=dict(bump_params.get('c_BE_params', BumpPopConfig().c_BE_params)),
        noise_scale_bump=bump_params.get('noise_scale_bump', BumpPopConfig.noise_scale_bump),
        sigma_B=bump_params.get('sigma_B', BumpPopConfig.sigma_B),
        sigma_I_BE=bump_params.get('sigma_I_BE', BumpPopConfig.sigma_I_BE),
        beta_B=bump_params.get('beta_B', BumpPopConfig.beta_B),
        clamp_frac_B=bump_params.get('clamp_frac_B', BumpPopConfig.clamp_frac_B),
        kernel_mode=bump_params.get('kernel_mode', BumpPopConfig.kernel_mode),
        kernel_gain=bump_params.get('kernel_gain', BumpPopConfig.kernel_gain),
        kernel_sigma=bump_params.get('kernel_sigma', BumpPopConfig.kernel_sigma),
        kernel_normed=bump_params.get('kernel_normed', BumpPopConfig.kernel_normed),
        be_kernel_mode=bump_params.get('be_kernel_mode', BumpPopConfig.be_kernel_mode),
        be_kernel_sigma=bump_params.get('be_kernel_sigma', BumpPopConfig.be_kernel_sigma),
        be_kernel_gain=bump_params.get('be_kernel_gain', BumpPopConfig.be_kernel_gain),
    )


def parse_decision_space_config(decision_space_params):
    return DecisionSpaceConfig(
        t_start=decision_space_params.get('t_start', DecisionSpaceConfig.t_start),
        dur1=decision_space_params.get('dur1', DecisionSpaceConfig.dur1),
        dur2=decision_space_params.get('dur2', DecisionSpaceConfig.dur2),
        boundary=decision_space_params.get('boundary', DecisionSpaceConfig.boundary),
        drift_rate=decision_space_params.get('drift_rate', DecisionSpaceConfig.drift_rate),
        noise_scale=decision_space_params.get('noise_scale', DecisionSpaceConfig.noise_scale),
        dt_DDM=decision_space_params.get('dt_DDM', DecisionSpaceConfig.dt_DDM),
        x0=decision_space_params.get('x0', DecisionSpaceConfig.x0),
        seed=decision_space_params.get('seed', DecisionSpaceConfig.seed),
    )


def parse_geometry_config(geometry_params, edge_config: EdgePopConfig, bump_config: BumpPopConfig):
    legacy_num_E = edge_config.legacy_num_E
    legacy_num_B = bump_config.legacy_num_B

    if legacy_num_E is not None and legacy_num_B is not None and legacy_num_E != legacy_num_B:
        raise ValueError("The number of neurons in the edge and bump population must be the same")

    if geometry_params is not None and 'num_units' in geometry_params:
        num_units = geometry_params['num_units']
    elif legacy_num_E is not None:
        num_units = legacy_num_E
    elif legacy_num_B is not None:
        num_units = legacy_num_B
    else:
        num_units = 1024

    if geometry_params is None:
        clamp_frac = edge_config.clamp_frac_E
        if not np.isclose(clamp_frac, bump_config.clamp_frac_B):
            raise ValueError("Legacy clamp fractions must match when shared geometry is implicit")
        return NeuralGeometryConfig(
            num_units=num_units,
            coding_limit=np.pi / 2,
            coding_frac=NeuralGeometryConfig.coding_frac,
            clamp_frac=clamp_frac,
        )

    coding_limit = geometry_params.get('coding_limit', np.pi / 2)
    coding_frac = geometry_params.get('coding_frac')
    clamp_frac = geometry_params.get('clamp_frac', edge_config.clamp_frac_E)
    parsed_num_units = geometry_params.get('num_units', num_units)

    if legacy_num_E is not None and parsed_num_units != legacy_num_E:
        raise ValueError("geometry.num_units must match legacy edge_pop.num_E")
    if legacy_num_B is not None and parsed_num_units != legacy_num_B:
        raise ValueError("geometry.num_units must match legacy bump_pop.num_B")

    if coding_frac is None:
        coding_frac = NeuralGeometryConfig.coding_frac

    return NeuralGeometryConfig(
        num_units=parsed_num_units,
        coding_limit=coding_limit,
        coding_frac=coding_frac,
        clamp_frac=clamp_frac,
    )


def build_geometry(config: NeuralGeometryConfig) -> DerivedGeometry:
    if config.num_units <= 0:
        raise ValueError("num_units must be positive")
    if config.coding_limit <= 0:
        raise ValueError("coding_limit must be positive")
    if config.coding_frac is None:
        raise ValueError("coding_frac must be specified")
    if not (0 < config.coding_frac <= 1):
        raise ValueError("coding_frac must be in (0, 1]")
    if not (0 <= config.clamp_frac < 1):
        raise ValueError("clamp_frac must be in [0, 1)")
    if config.coding_frac + config.clamp_frac > 1:
        raise ValueError("coding_frac + clamp_frac must be <= 1")

    clamp_side_width = int(config.num_units * config.clamp_frac / 2)
    coding_width = int(config.num_units * config.coding_frac)
    if coding_width <= 0:
        raise ValueError("coding_frac is too small for the chosen num_units")

    remaining = config.num_units - 2 * clamp_side_width - coding_width
    if remaining < 0:
        raise ValueError("Derived geometry is invalid: coding and clamp regions exceed population size")

    left_gap_width = remaining // 2
    right_gap_width = remaining - left_gap_width
    k1 = clamp_side_width + left_gap_width
    k2 = k1 + coding_width
    coding_theta_min = -config.coding_limit
    coding_theta_max = config.coding_limit
    full_theta_limit = config.coding_limit / config.coding_frac

    return DerivedGeometry(
        num_units=config.num_units,
        coding_limit=config.coding_limit,
        coding_frac=config.coding_frac,
        clamp_frac=config.clamp_frac,
        theta_min=-full_theta_limit,
        theta_max=full_theta_limit,
        coding_theta_min=coding_theta_min,
        coding_theta_max=coding_theta_max,
        clamp_side_width=clamp_side_width,
        left_gap_width=left_gap_width,
        right_gap_width=right_gap_width,
        coding_width=coding_width,
        k1=k1,
        k2=k2,
    )
