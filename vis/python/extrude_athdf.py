#!/usr/bin/env python3
"""Convert a uniform 2D cylindrical Athena++ athdf dump into a 3D IC file.

The output is consumed by ``pgen/z_pinch_3d_from_array.cpp``.  Hydrodynamic
variables and passive-scalar masses are restricted conservatively.  The
axisymmetric poloidal magnetic field is rebuilt from a discrete flux function,
so the face-centred field given to Athena++ has roundoff-level CT divergence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


@dataclass
class AthenaDump:
    path: Path
    time: float
    r_faces: np.ndarray
    theta_faces: np.ndarray
    z_faces: np.ndarray
    variables: dict[str, np.ndarray]
    axisymmetry_absolute_rms: dict[str, float]
    axisymmetry_rms: dict[str, float]


def _decode_strings(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            result.append(value.decode("ascii").rstrip("\x00"))
        else:
            result.append(str(value).rstrip("\x00"))
    return result


def _uniform_faces(root_grid: np.ndarray, count: int, label: str) -> np.ndarray:
    xmin, xmax, ratio = (float(x) for x in root_grid)
    if not np.isclose(ratio, 1.0, rtol=0.0, atol=1.0e-13):
        raise ValueError(
            f"{label} is nonuniform (RootGrid ratio={ratio}); this converter "
            "currently requires the uniform Task2a mesh"
        )
    return np.linspace(xmin, xmax, count + 1, dtype=np.float64)


def read_uniform_athdf(path: Path) -> AthenaDump:
    """Read selected primitive variables into global ``(z, theta, r)`` arrays."""

    with h5py.File(path, "r") as h5:
        coordinates = h5.attrs["Coordinates"]
        if isinstance(coordinates, bytes):
            coordinates = coordinates.decode("ascii").rstrip("\x00")
        if coordinates != "cylindrical":
            raise ValueError(f"expected cylindrical coordinates, found {coordinates!r}")

        levels = np.asarray(h5["Levels"])
        if np.any(levels != 0) or int(h5.attrs["MaxLevel"]) != 0:
            raise ValueError("SMR/AMR dumps are not supported; all blocks must be level 0")

        root_size = np.asarray(h5.attrs["RootGridSize"], dtype=int)
        block_size = np.asarray(h5.attrs["MeshBlockSize"], dtype=int)
        if np.any(root_size % block_size != 0):
            raise ValueError("root grid is not an integer number of source MeshBlocks")
        nr, ntheta, nz = (int(x) for x in root_size)
        br, btheta, bz = (int(x) for x in block_size)
        r_faces = _uniform_faces(h5.attrs["RootGridX1"], nr, "x1/r")
        theta_faces = _uniform_faces(h5.attrs["RootGridX2"], ntheta, "x2/theta")
        z_faces = _uniform_faces(h5.attrs["RootGridX3"], nz, "x3/z")

        dataset_names = _decode_strings(h5.attrs["DatasetNames"])
        variable_names = _decode_strings(h5.attrs["VariableNames"])
        num_variables = [int(x) for x in h5.attrs["NumVariables"]]
        variable_map: dict[str, tuple[str, int]] = {}
        offset = 0
        for dataset, count in zip(dataset_names, num_variables):
            for index, name in enumerate(variable_names[offset : offset + count]):
                variable_map[name] = (dataset, index)
            offset += count

        required = ("rho", "press", "vel1", "vel2", "vel3", "Bcc1", "Bcc2", "Bcc3")
        missing = [name for name in required if name not in variable_map]
        if missing:
            raise ValueError(f"athdf is missing required variables: {', '.join(missing)}")
        scalar_names = sorted(
            (name for name in variable_map if re.fullmatch(r"r\d+", name)),
            key=lambda name: int(name[1:]),
        )
        selected = required + tuple(scalar_names)
        variables = {
            name: np.empty((nz, ntheta, nr), dtype=np.float64) for name in selected
        }

        logical = np.asarray(h5["LogicalLocations"], dtype=np.int64)
        for block, (lx1, lx2, lx3) in enumerate(logical):
            si = slice(int(lx1) * br, (int(lx1) + 1) * br)
            sj = slice(int(lx2) * btheta, (int(lx2) + 1) * btheta)
            sk = slice(int(lx3) * bz, (int(lx3) + 1) * bz)
            for name in selected:
                dataset, index = variable_map[name]
                variables[name][sk, sj, si] = h5[dataset][index, block, :, :, :]

        dtheta = np.diff(theta_faces)
        theta_weight = dtheta[None, :, None]
        axisymmetry_absolute_rms: dict[str, float] = {}
        axisymmetry_rms: dict[str, float] = {}
        for name, values in variables.items():
            mean = np.sum(values * theta_weight, axis=1, keepdims=True) / np.sum(dtheta)
            numerator = np.sum((values - mean) ** 2 * theta_weight)
            denominator = np.sum(values**2 * theta_weight)
            axisymmetry_absolute_rms[name] = float(
                np.sqrt(numerator / (values.shape[0] * values.shape[2] * np.sum(dtheta)))
            )
            axisymmetry_rms[name] = float(
                np.sqrt(numerator / max(denominator, np.finfo(float).tiny))
            )

        return AthenaDump(
            path=path,
            time=float(h5.attrs["Time"]),
            r_faces=r_faces,
            theta_faces=theta_faces,
            z_faces=z_faces,
            variables=variables,
            axisymmetry_absolute_rms=axisymmetry_absolute_rms,
            axisymmetry_rms=axisymmetry_rms,
        )


def theta_average(values: np.ndarray, theta_faces: np.ndarray) -> np.ndarray:
    dtheta = np.diff(theta_faces)
    return np.sum(values * dtheta[None, :, None], axis=-2) / np.sum(dtheta)


def coarsen_weighted(
    values: np.ndarray,
    weights: np.ndarray,
    factor_z: int,
    factor_r: int,
) -> np.ndarray:
    """Volume-average a ``(z,r)`` cell array by integer factors."""

    nz, nr = values.shape
    if factor_z <= 0 or factor_r <= 0:
        raise ValueError("coarsening factors must be positive")
    if nz % factor_z or nr % factor_r:
        raise ValueError(
            f"source shape {(nz, nr)} is not divisible by factors "
            f"(z={factor_z}, r={factor_r})"
        )
    new_shape = (nz // factor_z, factor_z, nr // factor_r, factor_r)
    weighted = (values * weights).reshape(new_shape).sum(axis=(1, 3))
    weight_sum = weights.reshape(new_shape).sum(axis=(1, 3))
    return weighted / weight_sum


def cylindrical_cell_weights(r_faces: np.ndarray, z_faces: np.ndarray) -> np.ndarray:
    radial_area = 0.5 * np.diff(r_faces**2)
    return np.diff(z_faces)[:, None] * radial_area[None, :]


def reconstruct_poloidal_field(
    br_cell: np.ndarray,
    bz_cell: np.ndarray,
    r_faces: np.ndarray,
    z_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Reconstruct a periodic, exactly divergence-free axisymmetric poloidal field.

    A discrete flux function ``psi(z_face, r_face)`` is integrated from the
    radial field.  Its z-independent radial offsets are then chosen to retain
    the axial mean of Bz.  This is the cylindrical CT analogue of taking a curl.
    """

    nz, nr = br_cell.shape
    if bz_cell.shape != (nz, nr):
        raise ValueError("Br and Bz shapes differ")
    if len(r_faces) != nr + 1 or len(z_faces) != nz + 1:
        raise ValueError("face arrays do not match cell arrays")
    dz = np.diff(z_faces)
    if not np.allclose(dz, dz[0], rtol=1.0e-12, atol=1.0e-14):
        raise ValueError("the flux reconstruction currently requires uniform z")

    r_volume = (2.0 / 3.0) * np.diff(r_faces**3) / np.diff(r_faces**2)
    br_target_face = np.zeros((nz, nr + 1), dtype=np.float64)
    if nr > 1:
        delta = r_volume[1:] - r_volume[:-1]
        left_weight = (r_volume[1:] - r_faces[1:-1]) / delta
        right_weight = (r_faces[1:-1] - r_volume[:-1]) / delta
        br_target_face[:, 1:-1] = (
            left_weight[None, :] * br_cell[:, :-1]
            + right_weight[None, :] * br_cell[:, 1:]
        )

    removed_mean = np.mean(br_target_face, axis=0)
    br_target_face -= removed_mean[None, :]

    psi = np.zeros((nz + 1, nr + 1), dtype=np.float64)
    for k in range(nz):
        psi[k + 1, :] = (
            psi[k, :] - r_faces * dz[k] * br_target_face[k, :]
        )
    periodic_closure_before = float(np.max(np.abs(psi[-1, :] - psi[0, :])))
    psi[-1, :] = psi[0, :]

    radial_area = 0.5 * np.diff(r_faces**2)
    offsets = np.zeros(nr + 1, dtype=np.float64)
    for i in range(nr):
        base_difference = np.mean(psi[:-1, i + 1] - psi[:-1, i])
        offsets[i + 1] = (
            offsets[i] + radial_area[i] * np.mean(bz_cell[:, i]) - base_difference
        )
    psi += offsets[None, :]

    br_face = -(psi[1:, :] - psi[:-1, :]) / (dz[:, None] * r_faces[None, :])
    bz_face = (psi[:, 1:] - psi[:, :-1]) / radial_area[None, :]

    left_weight = (r_faces[1:] - r_volume) / np.diff(r_faces)
    right_weight = (r_volume - r_faces[:-1]) / np.diff(r_faces)
    br_reconstructed = (
        left_weight[None, :] * br_face[:, :-1]
        + right_weight[None, :] * br_face[:, 1:]
    )
    bz_reconstructed = 0.5 * (bz_face[:-1, :] + bz_face[1:, :])

    volume = radial_area[None, :] * dz[:, None]
    radial_flux_difference = dz[:, None] * (
        r_faces[None, 1:] * br_face[:, 1:]
        - r_faces[None, :-1] * br_face[:, :-1]
    )
    axial_flux_difference = radial_area[None, :] * (
        bz_face[1:, :] - bz_face[:-1, :]
    )
    divergence = (radial_flux_difference + axial_flux_difference) / volume

    target_norm = np.sqrt(np.mean(br_cell**2 + bz_cell**2))
    mismatch_norm = np.sqrt(
        np.mean((br_reconstructed - br_cell) ** 2 + (bz_reconstructed - bz_cell) ** 2)
    )
    diagnostics = {
        "max_abs_divergence": float(np.max(np.abs(divergence))),
        "poloidal_cell_rms": float(target_norm),
        "poloidal_reconstruction_rms": float(mismatch_norm),
        "poloidal_reconstruction_relative_rms": float(
            mismatch_norm / max(target_norm, np.finfo(float).tiny)
        ),
        "max_removed_mean_Br": float(np.max(np.abs(removed_mean))),
        "periodic_psi_closure_before_fix": periodic_closure_before,
    }
    return br_face, bz_face, diagnostics


def parse_integer_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def read_athinput(path: Path) -> dict[str, dict[str, str]]:
    """Read the simple ``<block>/key=value`` subset used by Athena inputs."""

    result: dict[str, dict[str, str]] = {}
    block: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("<") and ">" in line:
            block = line[1 : line.index(">")].strip()
            result.setdefault(block, {})
            continue
        if block is not None and "=" in line:
            key, value = line.split("=", 1)
            result[block][key.strip()] = value.strip()
    return result


def athinput_float(
    values: dict[str, dict[str, str]],
    block: str,
    *keys: str,
) -> float | None:
    for key in keys:
        if key in values.get(block, {}):
            return float(values[block][key])
    return None


def make_seed_pattern(
    r_centres: np.ndarray,
    theta_centres: np.ndarray,
    z_centres: np.ndarray,
    r_faces: np.ndarray,
    z_faces: np.ndarray,
    modes_m: tuple[int, ...],
    modes_kz: tuple[int, ...],
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    pattern = np.zeros(
        (len(z_centres), len(theta_centres), len(r_centres)), dtype=np.float64
    )
    for mode_m in modes_m:
        for mode_kz in modes_kz:
            phase_theta, phase_z = rng.uniform(0.0, 2.0 * np.pi, size=2)
            angular = np.cos(mode_m * theta_centres + phase_theta)
            axial = np.cos(
                2.0
                * np.pi
                * mode_kz
                * (z_centres - z_faces[0])
                / (z_faces[-1] - z_faces[0])
                + phase_z
            )
            pattern += axial[:, None, None] * angular[None, :, None]

    radial_envelope = np.sin(
        np.pi * (r_centres - r_faces[0]) / (r_faces[-1] - r_faces[0])
    )
    pattern *= radial_envelope[None, None, :]
    rms = float(np.sqrt(np.mean(pattern**2)))
    if rms == 0.0:
        raise ValueError("seed pattern has zero RMS")
    return pattern / rms


def _cell_centred_from_faces(
    br_face: np.ndarray,
    btheta_cell: np.ndarray,
    bz_face: np.ndarray,
    r_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_volume = (2.0 / 3.0) * np.diff(r_faces**3) / np.diff(r_faces**2)
    left_weight = (r_faces[1:] - r_volume) / np.diff(r_faces)
    right_weight = (r_volume - r_faces[:-1]) / np.diff(r_faces)
    br_cell = (
        left_weight[None, :] * br_face[:, :-1]
        + right_weight[None, :] * br_face[:, 1:]
    )
    bz_cell = 0.5 * (bz_face[:-1, :] + bz_face[1:, :])
    return br_cell, btheta_cell, bz_cell


def _pack_blocks(
    values: np.ndarray,
    block_z: int,
    block_theta: int,
    block_r: int,
    face_direction: int | None = None,
) -> np.ndarray:
    """Pack global arrays in lexicographic ``(lx3,lx2,lx1)`` block order."""

    prefix = values.shape[:-3]
    nz = values.shape[-3] - (1 if face_direction == 3 else 0)
    ntheta = values.shape[-2] - (1 if face_direction == 2 else 0)
    nr = values.shape[-1] - (1 if face_direction == 1 else 0)
    if nz % block_z or ntheta % block_theta or nr % block_r:
        raise ValueError("target mesh is not divisible by requested MeshBlock sizes")
    nbz, nbt, nbr = nz // block_z, ntheta // block_theta, nr // block_r
    nblocks = nbz * nbt * nbr
    local_shape = (
        block_z + (1 if face_direction == 3 else 0),
        block_theta + (1 if face_direction == 2 else 0),
        block_r + (1 if face_direction == 1 else 0),
    )
    packed = np.empty(prefix + (nblocks,) + local_shape, dtype=np.float64)
    block = 0
    for lk in range(nbz):
        for lj in range(nbt):
            for li in range(nbr):
                sk = slice(lk * block_z, (lk + 1) * block_z + (face_direction == 3))
                sj = slice(
                    lj * block_theta,
                    (lj + 1) * block_theta + (face_direction == 2),
                )
                si = slice(li * block_r, (li + 1) * block_r + (face_direction == 1))
                packed[..., block, :, :, :] = values[..., sk, sj, si]
                block += 1
    return packed


def build_initial_condition(
    source: AthenaDump,
    *,
    gamma: float,
    coarsen_r: int,
    coarsen_z: int,
    ntheta: int,
    block_r: int,
    block_theta: int,
    block_z: int,
    noise_amplitude: float,
    seed_modes: tuple[int, ...],
    seed_kz: tuple[int, ...],
    random_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if gamma <= 1.0:
        raise ValueError("gamma must exceed 1")
    if ntheta < 4:
        raise ValueError("a 3D m=1 test needs at least four theta cells")
    if max(seed_modes) > ntheta // 2:
        raise ValueError(
            f"requested m={max(seed_modes)} exceeds the theta Nyquist mode {ntheta // 2}"
        )
    source_nz = len(source.z_faces) - 1
    target_nz = source_nz // coarsen_z
    if max(seed_kz) > target_nz // 2:
        raise ValueError(
            f"requested kz={max(seed_kz)} exceeds the axial Nyquist mode {target_nz // 2}"
        )
    if noise_amplitude < 0.0:
        raise ValueError("noise amplitude cannot be negative")

    v = source.variables
    rho3 = v["rho"]
    pressure3 = v["press"]
    velocity3 = np.stack((v["vel1"], v["vel2"], v["vel3"]))
    magnetic3 = np.stack((v["Bcc1"], v["Bcc2"], v["Bcc3"]))
    energy3 = (
        pressure3 / (gamma - 1.0)
        + 0.5 * rho3 * np.sum(velocity3**2, axis=0)
        + 0.5 * np.sum(magnetic3**2, axis=0)
    )
    conserved3 = np.concatenate(
        (
            rho3[None, ...],
            rho3[None, ...] * velocity3,
            energy3[None, ...],
        ),
        axis=0,
    )
    scalar_names = sorted(
        (name for name in v if re.fullmatch(r"r\d+", name)),
        key=lambda name: int(name[1:]),
    )
    scalars3 = np.stack([rho3 * v[name] for name in scalar_names])

    weights_fine = cylindrical_cell_weights(source.r_faces, source.z_faces)
    conserved2 = np.stack(
        [
            coarsen_weighted(
                theta_average(field, source.theta_faces),
                weights_fine,
                coarsen_z,
                coarsen_r,
            )
            for field in conserved3
        ]
    )
    scalars2 = np.stack(
        [
            coarsen_weighted(
                theta_average(field, source.theta_faces),
                weights_fine,
                coarsen_z,
                coarsen_r,
            )
            for field in scalars3
        ]
    )
    magnetic2 = np.stack(
        [
            coarsen_weighted(
                theta_average(field, source.theta_faces),
                weights_fine,
                coarsen_z,
                coarsen_r,
            )
            for field in magnetic3
        ]
    )

    r_faces = source.r_faces[::coarsen_r].copy()
    z_faces = source.z_faces[::coarsen_z].copy()
    if r_faces[-1] != source.r_faces[-1]:
        r_faces = np.append(r_faces, source.r_faces[-1])
    if z_faces[-1] != source.z_faces[-1]:
        z_faces = np.append(z_faces, source.z_faces[-1])
    theta_faces = np.linspace(
        source.theta_faces[0], source.theta_faces[-1], ntheta + 1, dtype=np.float64
    )
    r_centres = (2.0 / 3.0) * np.diff(r_faces**3) / np.diff(r_faces**2)
    theta_centres = 0.5 * (theta_faces[:-1] + theta_faces[1:])
    z_centres = 0.5 * (z_faces[:-1] + z_faces[1:])

    br_face2, bz_face2, magnetic_diagnostics = reconstruct_poloidal_field(
        magnetic2[0], magnetic2[2], r_faces, z_faces
    )
    br_cell2, btheta_cell2, bz_cell2 = _cell_centred_from_faces(
        br_face2, magnetic2[1], bz_face2, r_faces
    )
    reconstructed_b2 = br_cell2**2 + btheta_cell2**2 + bz_cell2**2

    rho2 = conserved2[0]
    velocity2 = conserved2[1:4] / rho2[None, :, :]
    pressure2 = (gamma - 1.0) * (
        conserved2[4]
        - 0.5 * np.sum(conserved2[1:4] ** 2, axis=0) / rho2
        - 0.5 * reconstructed_b2
    )
    if not np.all(np.isfinite(pressure2)) or np.min(pressure2) <= 0.0:
        raise ValueError(
            "conservative restriction plus magnetic reconstruction produced "
            f"non-positive pressure (minimum {np.min(pressure2):.6e})"
        )

    cons_global = np.repeat(conserved2[:, :, None, :], ntheta, axis=2)
    scalar_global = np.repeat(scalars2[:, :, None, :], ntheta, axis=2)
    br_global = np.repeat(br_face2[:, None, :], ntheta, axis=1)
    btheta_global = np.repeat(magnetic2[1, :, None, :], ntheta + 1, axis=1)
    bz_global = np.repeat(bz_face2[:, None, :], ntheta, axis=1)

    pattern = make_seed_pattern(
        r_centres,
        theta_centres,
        z_centres,
        r_faces,
        z_faces,
        seed_modes,
        seed_kz,
        random_seed,
    )
    fast_speed = np.sqrt(
        gamma * pressure2 / rho2 + reconstructed_b2 / rho2
    )
    delta_vr = noise_amplitude * fast_speed[:, None, :] * pattern
    old_momentum = cons_global[1].copy()
    cons_global[1] += cons_global[0] * delta_vr
    cons_global[4] += (
        (cons_global[1] ** 2 - old_momentum**2) / (2.0 * cons_global[0])
    )

    datasets = {
        "cons": _pack_blocks(
            cons_global, block_z, block_theta, block_r, face_direction=None
        ),
        "scalars": _pack_blocks(
            scalar_global, block_z, block_theta, block_r, face_direction=None
        ),
        "b1": _pack_blocks(
            br_global, block_z, block_theta, block_r, face_direction=1
        ),
        "b2": _pack_blocks(
            btheta_global, block_z, block_theta, block_r, face_direction=2
        ),
        "b3": _pack_blocks(
            bz_global, block_z, block_theta, block_r, face_direction=3
        ),
    }

    fine_integral = np.sum(
        theta_average(conserved3, source.theta_faces)
        * weights_fine[None, :, :]
        * (source.theta_faces[-1] - source.theta_faces[0]),
        axis=(1, 2),
    )
    weights_coarse = cylindrical_cell_weights(r_faces, z_faces)
    coarse_integral_before_seed = np.sum(
        conserved2
        * weights_coarse[None, :, :]
        * (theta_faces[-1] - theta_faces[0]),
        axis=(1, 2),
    )
    relative_conservation = (
        (coarse_integral_before_seed - fine_integral)
        / np.maximum(np.abs(fine_integral), np.finfo(float).tiny)
    )

    diagnostics: dict[str, object] = {
        "source": str(source.path),
        "source_time": source.time,
        "source_shape_r_theta_z": [
            len(source.r_faces) - 1,
            len(source.theta_faces) - 1,
            len(source.z_faces) - 1,
        ],
        "target_shape_r_theta_z": [len(r_centres), ntheta, len(z_centres)],
        "meshblock_shape_r_theta_z": [block_r, block_theta, block_z],
        "coarsen_r": coarsen_r,
        "coarsen_z": coarsen_z,
        "gamma": gamma,
        "number_of_scalars": len(scalar_names),
        "scalar_names": scalar_names,
        "axisymmetry_absolute_rms": source.axisymmetry_absolute_rms,
        "axisymmetry_relative_rms": source.axisymmetry_rms,
        "conserved_integrals_source": fine_integral.tolist(),
        "conserved_integrals_restricted_before_seed": coarse_integral_before_seed.tolist(),
        "conserved_integral_relative_error": relative_conservation.tolist(),
        "minimum_pressure_after_restriction": float(np.min(pressure2)),
        "maximum_pressure_after_restriction": float(np.max(pressure2)),
        "noise_amplitude_fast_speed_fraction": noise_amplitude,
        "seed_modes_m": list(seed_modes) if noise_amplitude > 0.0 else [],
        "seed_modes_kz_integer": list(seed_kz) if noise_amplitude > 0.0 else [],
        "random_seed": random_seed if noise_amplitude > 0.0 else None,
        "seed_delta_vr_rms": float(np.sqrt(np.mean(delta_vr**2))),
        "seed_delta_vr_max_abs": float(np.max(np.abs(delta_vr))),
        **magnetic_diagnostics,
    }
    return datasets, {
        "diagnostics": diagnostics,
        "r_faces": r_faces,
        "theta_faces": theta_faces,
        "z_faces": z_faces,
        "rho2": rho2,
        "pressure2": pressure2,
        "btheta2": magnetic2[1],
    }


def write_ic(
    output: Path,
    datasets: dict[str, np.ndarray],
    metadata: dict[str, object],
    *,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    diagnostics = metadata["diagnostics"]
    with h5py.File(output, "w") as h5:
        for name, values in datasets.items():
            h5.create_dataset(name, data=values, dtype=np.float64)
        h5.attrs["format"] = "athena++ driven z-pinch extruded IC v1"
        h5.attrs["coordinates"] = "cylindrical"
        h5.attrs["source_athdf"] = str(diagnostics["source"])
        h5.attrs["source_time"] = float(diagnostics["source_time"])
        h5.attrs["gamma"] = float(diagnostics["gamma"])
        h5.attrs["root_grid_size"] = np.asarray(
            diagnostics["target_shape_r_theta_z"], dtype=np.int64
        )
        h5.attrs["meshblock_size"] = np.asarray(
            diagnostics["meshblock_shape_r_theta_z"], dtype=np.int64
        )
        h5.attrs["number_of_scalars"] = int(diagnostics["number_of_scalars"])
        h5.attrs["diagnostics_json"] = json.dumps(diagnostics, sort_keys=True)


def write_athinput(
    path: Path,
    ic_path: Path,
    metadata: dict[str, object],
    *,
    source_input: dict[str, dict[str, str]],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    diagnostics = metadata["diagnostics"]
    r_faces = np.asarray(metadata["r_faces"])
    theta_faces = np.asarray(metadata["theta_faces"])
    z_faces = np.asarray(metadata["z_faces"])
    nr, ntheta, nz = diagnostics["target_shape_r_theta_z"]
    block_r, block_theta, block_z = diagnostics["meshblock_shape_r_theta_z"]
    gamma = float(diagnostics["gamma"])

    problem_id = re.sub(r"[^A-Za-z0-9_]+", "_", ic_path.stem).strip("_")

    def copied(block: str) -> dict[str, str]:
        return dict(source_input.get(block, {}))

    blocks: list[tuple[str, dict[str, str]]] = []
    blocks.append(("job", {"problem_id": problem_id}))
    for name in source_input:
        if name.startswith("output"):
            blocks.append((name, copied(name)))

    time_values = copied("time")
    time_values.pop("start_time", None)
    blocks.append(("time", time_values))

    mesh_values = copied("mesh")
    mesh_values.update(
        {
            "nx1": str(nr),
            "x1min": f"{r_faces[0]:.17g}",
            "x1max": f"{r_faces[-1]:.17g}",
            "nx2": str(ntheta),
            "x2min": f"{theta_faces[0]:.17g}",
            "x2max": f"{theta_faces[-1]:.17g}",
            "nx3": str(nz),
            "x3min": f"{z_faces[0]:.17g}",
            "x3max": f"{z_faces[-1]:.17g}",
            "refinement": "none",
        }
    )
    blocks.append(("mesh", mesh_values))
    blocks.append(
        (
            "meshblock",
            {"nx1": str(block_r), "nx2": str(block_theta), "nx3": str(block_z)},
        )
    )

    hydro_values = copied("hydro")
    hydro_values["gamma"] = f"{gamma:.17g}"
    blocks.append(("hydro", hydro_values))

    problem_values = copied("problem")
    problem_values.update(
        {
            "input_filename": str(ic_path.resolve()),
            "dataset_cons": "cons",
            "dataset_scalars": "scalars",
            "dataset_b1": "b1",
            "dataset_b2": "b2",
            "dataset_b3": "b3",
        }
    )
    blocks.append(("problem", problem_values))

    emitted = {name for name, _ in blocks}
    for name, values in source_input.items():
        if name not in emitted and name not in {"comment"}:
            blocks.append((name, dict(values)))

    lines = [
        "<comment>",
        f"problem = driven 3D continuation extruded from {diagnostics['source']}",
        f"source_time = {diagnostics['source_time']}",
        (
            "configure = --prob=z_pinch_heated --coord=cylindrical "
            f"--eos=adiabatic --flux=hlld --nscalars={diagnostics['number_of_scalars']} "
            "-b -mpi -hdf5 -h5double (with z_pinch_heated_from_array.patch)"
        ),
        "",
    ]
    for block, values in blocks:
        lines.append(f"<{block}>")
        lines.extend(f"{key} = {value}" for key, value in values.items())
        lines.append("")
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively coarsen a 2D cylindrical Athena++ athdf dump, "
            "extrude it in theta, and write a face-field 3D initial condition "
            "plus a fully driven z_pinch_heated athinput."
        )
    )
    parser.add_argument("source", type=Path, help="2D primitive athdf dump")
    parser.add_argument(
        "--source-athinput",
        type=Path,
        required=True,
        help=(
            "original 2D athinput; all forcing, controller, tracer, boundary, "
            "output, and time parameters are carried into the 3D input"
        ),
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="output IC HDF5")
    parser.add_argument("--athinput", type=Path, help="companion athinput path")
    parser.add_argument("--report", type=Path, help="JSON diagnostics path")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--coarsen-r", type=int, default=2)
    parser.add_argument("--coarsen-z", type=int, default=2)
    parser.add_argument("--ntheta", type=int, default=8)
    parser.add_argument("--block-r", type=int, default=32)
    parser.add_argument("--block-theta", type=int, default=4)
    parser.add_argument("--block-z", type=int, default=60)
    parser.add_argument(
        "--noise-amplitude",
        type=float,
        default=0.0,
        help="optional controlled perturbation; production default is exactly zero",
    )
    parser.add_argument("--seed-modes", type=parse_integer_list, default=(1,))
    parser.add_argument("--seed-kz", type=parse_integer_list, default=(1, 2, 3, 4))
    parser.add_argument("--random-seed", type=int, default=20260724)
    parser.add_argument("--inner-density", type=float)
    parser.add_argument("--inner-pressure", type=float)
    parser.add_argument("--bin", dest="bin_value", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    source_input: dict[str, dict[str, str]] = {}
    if args.source_athinput is not None:
        source_input = read_athinput(args.source_athinput)
    gamma = args.gamma
    if gamma is None:
        gamma = athinput_float(source_input, "hydro", "gamma")
    if gamma is None:
        gamma = 5.0 / 3.0
    inner_density = args.inner_density
    if inner_density is None:
        inner_density = athinput_float(
            source_input, "problem", "inner_bc_dens", "densmin"
        )
    inner_pressure = args.inner_pressure
    if inner_pressure is None:
        inner_pressure = athinput_float(
            source_input, "problem", "inner_bc_pres", "presmin"
        )
    bin_value = args.bin_value
    if bin_value is None:
        bin_value = athinput_float(source_input, "problem", "bin")

    source = read_uniform_athdf(args.source)
    datasets, metadata = build_initial_condition(
        source,
        gamma=gamma,
        coarsen_r=args.coarsen_r,
        coarsen_z=args.coarsen_z,
        ntheta=args.ntheta,
        block_r=args.block_r,
        block_theta=args.block_theta,
        block_z=args.block_z,
        noise_amplitude=args.noise_amplitude,
        seed_modes=args.seed_modes,
        seed_kz=args.seed_kz,
        random_seed=args.random_seed,
    )
    boundary_provenance: dict[str, str] = {}
    if inner_density is None:
        inner_density = float(np.median(np.asarray(metadata["rho2"])[:, 0]))
        boundary_provenance["inner_bc_dens"] = "inferred from first active radial cell"
    else:
        boundary_provenance["inner_bc_dens"] = (
            "explicit CLI" if args.inner_density is not None else "source athinput"
        )
    if inner_pressure is None:
        inner_pressure = float(np.median(np.asarray(metadata["pressure2"])[:, 0]))
        boundary_provenance["inner_bc_pres"] = "inferred from first active radial cell"
    else:
        boundary_provenance["inner_bc_pres"] = (
            "explicit CLI" if args.inner_pressure is not None else "source athinput"
        )
    if bin_value is None:
        r_faces = np.asarray(metadata["r_faces"])
        r_centres = (2.0 / 3.0) * np.diff(r_faces**3) / np.diff(r_faces**2)
        btheta2 = np.asarray(metadata["btheta2"])
        bin_value = float(np.median(btheta2[:, 0] * r_centres[0]) / r_faces[0])
        boundary_provenance["bin"] = "inferred from first active radial cell"
    else:
        boundary_provenance["bin"] = (
            "explicit CLI" if args.bin_value is not None else "source athinput"
        )
    metadata["diagnostics"]["radial_boundary_parameters"] = {
        "inner_bc_dens": inner_density,
        "inner_bc_pres": inner_pressure,
        "bin": bin_value,
        "provenance": boundary_provenance,
    }
    metadata["diagnostics"]["source_athinput"] = str(args.source_athinput)
    metadata["diagnostics"]["evolution"] = (
        "full forced z_pinch_heated evolution; only ProblemGenerator initialization "
        "is replaced by the extruded array"
    )
    metadata["diagnostics"]["copied_problem_parameters"] = dict(
        source_input.get("problem", {})
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_ic(args.output, datasets, metadata, overwrite=args.overwrite)

    athinput_path = args.athinput or args.output.with_suffix(".athinput")
    write_athinput(
        athinput_path,
        args.output,
        metadata,
        source_input=source_input,
        overwrite=args.overwrite,
    )
    report_path = args.report or args.output.with_suffix(".json")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"{report_path} exists; pass --overwrite to replace it")
    report_path.write_text(
        json.dumps(metadata["diagnostics"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata["diagnostics"], indent=2, sort_keys=True))
    print(f"wrote IC:       {args.output}")
    print(f"wrote athinput: {athinput_path}")
    print(f"wrote report:   {report_path}")


if __name__ == "__main__":
    main()
