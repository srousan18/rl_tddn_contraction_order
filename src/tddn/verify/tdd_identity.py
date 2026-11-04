"""Identity check for the fully contracted TDD of a miter.

The miter of an equivalence check is ``C . (C')^-1``. If ``C == C'`` it is the
identity up to a global phase, so a final TDD that is not identity-like
proves non-equivalence.

The checks must run in the TDD session that built the final TDD: the
reference identity uses indices from the session's global index order.
"""

from __future__ import annotations

from functools import reduce
from typing import Iterable, Sequence

import numpy as np

from tddn.backend.TDD import Index, cont, get_identity_tdd, get_tdd

# Building the dense operator costs 4**n complex entries; refuse beyond this.
DEFAULT_DENSE_MAX_QUBITS = 10

# Status values reported by check_final_tdd().
STATUS_IDENTITY = "identity"
STATUS_NOT_IDENTITY = "not_identity"
STATUS_INCOMPLETE = "incomplete"
STATUS_LABEL_MISMATCH = "label_mismatch"
STATUS_UNAVAILABLE = "unavailable"


def tdd_axis_labels(tdd) -> tuple[str, ...]:
    """Open-leg labels of ``tdd``, ordered from the top key downwards."""

    if not tdd.key_2_index:
        return ()
    keys = [k for k in tdd.key_2_index if k != -1]
    if not keys:
        return ()
    return tuple(tdd.key_2_index[k] for k in range(max(keys), -1, -1))


def qubits_from_labels(labels: Iterable[str]) -> list[int]:
    """Qubit numbers appearing as ``x<q>``/``y<q>`` among ``labels``."""

    return sorted({
        int(label[1:])
        for label in labels
        if len(label) > 1 and label[0] in {"x", "y"} and label[1:].isdigit()
    })


def _has_full_xy_legs(labels: Sequence[str]) -> bool:
    """True when ``labels`` is exactly one ``x<q>`` and one ``y<q>`` per qubit."""

    label_set = set(labels)
    if len(label_set) != len(labels):
        return False
    qubits = qubits_from_labels(label_set)
    expected = {f"x{q}" for q in qubits} | {f"y{q}" for q in qubits}
    return bool(qubits) and label_set == expected


def identity_tdd_for_labels(labels: Iterable[str]):
    """Identity TDD carrying exactly the ``x<q>``/``y<q>`` legs in ``labels``.

    Built from the final TDD's own labels, since idle qubits may have no x/y
    index in the session. ``None`` when the legs are not one x/y pair per
    qubit or an index is unknown to the session.
    """

    labels = set(labels)
    qubits = qubits_from_labels(labels)
    if not qubits:
        return None

    result = get_identity_tdd()
    eye = np.eye(2, dtype=np.complex128)
    for q in qubits:
        x_label, y_label = f"x{q}", f"y{q}"
        if x_label not in labels or y_label not in labels:
            return None
        try:
            tensor = get_tdd(eye, [Index(x_label), Index(y_label)])
            result = cont(result, tensor)
        except KeyError:
            # Index absent from this session's global order.
            return None
    return result


def nodes_structurally_equal(a, b, atol: float = 1e-6) -> bool:
    """Compare two TDD sub-DAGs by structure rather than by object identity.

    The environment deep-copies its TDD table on reset, so nodes of a
    contracted TDD are not shared with a freshly built reference even when
    they are structurally identical. Both TDDs are normalized the same way, so
    equal operators (up to the root weight) have the same shape and weights.
    """

    memo: dict[tuple[int, int], bool] = {}

    def walk(x, y) -> bool:
        if x is y:
            return True
        if x is None or y is None:
            return False
        cache_key = (id(x), id(y))
        cached = memo.get(cache_key)
        if cached is not None:
            return cached

        # Assume equal while recursing: the DAG is acyclic, so a back-reference
        # can only be reached through a pair already being proven.
        memo[cache_key] = True
        result = True

        if x.key != y.key or x.succ_num != y.succ_num:
            result = False
        else:
            for k in range(x.succ_num):
                if abs(complex(x.out_weight[k]) - complex(y.out_weight[k])) > atol:
                    result = False
                    break
                if not walk(x.successor[k], y.successor[k]):
                    result = False
                    break

        memo[cache_key] = result
        return result

    return walk(a, b)


def compare_up_to_global_phase(actual, expected, atol: float = 1e-6,
                               weight_atol: float = 1e-4) -> tuple[bool, float]:
    """Compare two TDDs ignoring a global phase.

    The node structure must match; only the root weights may differ, and equality
    up to global phase means their ratio has unit magnitude.  Returns
    ``(matches, fidelity_like)`` where ``fidelity_like`` is ``1 - |magnitude
    error|``, clamped at 0.

    ``weight_atol`` is looser than ``atol`` because the root weight
    accumulates rounding over every contraction and drifts with circuit size;
    the structure, compared with ``atol``, still has to match exactly.
    """

    if tdd_axis_labels(actual) != tdd_axis_labels(expected):
        return False, 0.0

    if not nodes_structurally_equal(actual.node, expected.node, atol=atol):
        return False, 0.0

    atol = weight_atol
    expected_weight = complex(expected.weight)
    actual_weight = complex(actual.weight)
    if abs(expected_weight) < atol:
        matches = abs(actual_weight) < atol
        return matches, 1.0 if matches else 0.0

    ratio = actual_weight / expected_weight
    magnitude_error = abs(abs(ratio) - 1.0)
    fidelity_like = max(0.0, 1.0 - float(magnitude_error))
    return bool(magnitude_error <= atol), fidelity_like


def check_identity_amplitudes(tdd, samples: int = 32, atol: float = 1e-4,
                              seed: int = 20260730) -> tuple[bool, float]:
    """Identity check by sampling amplitudes; robust to canonicalization drift.

    TDD reduction merges nodes only when weights agree to within ``epi``
    (1e-6), so accumulated rounding can leave an equivalent tensor with a
    different DAG shape. Sampling amplitudes does not depend on the shape: for
    the identity, ``U[b] = 1`` when every x_q equals its y_q and ``0``
    otherwise, up to the global phase in the root weight. Diagonal and
    off-diagonal entries are both sampled.

    Returns ``(matches, worst_absolute_deviation)``.
    """

    labels = list(tdd_axis_labels(tdd))
    if not _has_full_xy_legs(labels):
        return False, float("inf")

    qubits = qubits_from_labels(labels)
    pos = {label: i for i, label in enumerate(labels)}
    rng = np.random.default_rng(seed)

    phase = complex(tdd.weight)
    if abs(phase) > 1e-12:
        phase /= abs(phase)
    else:
        phase = 1.0 + 0.0j

    worst = 0.0
    for s in range(samples):
        bits = [0] * len(labels)
        assign = {q: int(rng.integers(0, 2)) for q in qubits}
        for q in qubits:
            bits[pos[f"x{q}"]] = assign[q]
            bits[pos[f"y{q}"]] = assign[q]
        expected = 1.0
        # Half the samples probe off-diagonal entries, which must vanish.
        if s % 2 == 1:
            flip = qubits[int(rng.integers(0, len(qubits)))]
            bits[pos[f"y{flip}"]] ^= 1
            expected = 0.0

        amp = complex(tdd.get_amplitude(list(bits))) / phase
        worst = max(worst, abs(amp - expected))

    return bool(worst <= atol), worst


def check_identity(tdd, atol: float = 1e-6) -> tuple[bool, float]:
    """TDD-native identity check: no dense operator is materialized.

    Tries exact structural comparison first, then amplitude sampling
    (:func:`check_identity_amplitudes`).
    """

    identity = identity_tdd_for_labels(tdd_axis_labels(tdd))
    if identity is not None:
        matches, fidelity = compare_up_to_global_phase(tdd, identity, atol=atol)
        if matches:
            return True, fidelity

    matches, worst = check_identity_amplitudes(tdd)
    return matches, max(0.0, 1.0 - worst) if matches else 0.0


def identity_tensor(axis_labels: Sequence[str]) -> np.ndarray | None:
    """Dense identity operator with axes ordered as ``axis_labels``.

    Built by Kronecker product in the canonical ``x0..x_{n-1}, y0..y_{n-1}``
    order and then transposed, which avoids a 4**n Python loop.
    """

    labels = list(axis_labels)
    if not _has_full_xy_legs(labels):
        return None

    qubits = qubits_from_labels(labels)
    n = len(qubits)
    canonical = reduce(np.kron, [np.eye(2, dtype=np.complex128)] * n)
    canonical = canonical.reshape((2,) * (2 * n))
    canonical_labels = [f"x{q}" for q in qubits] + [f"y{q}" for q in qubits]

    perm = [canonical_labels.index(label) for label in labels]
    return np.transpose(canonical, perm)


def global_phase(mat: np.ndarray, target: np.ndarray) -> complex:
    """Unit-modulus phase best relating ``mat`` to ``target``."""

    mask = np.abs(target) > 1e-12
    if not np.any(mask):
        return 1.0 + 0.0j
    ratios = mat[mask] / target[mask]
    idx = int(np.argmax(np.abs(ratios)))
    phase = ratios.reshape(-1)[idx]
    magnitude = abs(phase)
    if magnitude < 1e-12:
        return 1.0 + 0.0j
    return phase / magnitude


def check_identity_dense(tdd, atol: float = 1e-6) -> tuple[bool, float]:
    """Dense cross-check of :func:`check_identity`; small qubit counts only.

    Materializes the 4**n operator, so the caller is responsible for gating on
    qubit count (see ``DEFAULT_DENSE_MAX_QUBITS``).
    """

    axis_labels = tdd_axis_labels(tdd)
    target = identity_tensor(axis_labels)
    if target is None:
        return False, 0.0

    arr = np.asarray(tdd.to_array(), dtype=np.complex128)
    if arr.size != target.size:
        return False, 0.0
    arr = arr.reshape(target.shape)

    phase = global_phase(arr, target)
    diff = float(np.linalg.norm(arr - phase * target))
    denom = max(1.0, float(np.linalg.norm(target)))
    fidelity_like = max(0.0, 1.0 - diff / denom)
    return bool(diff <= atol * denom), fidelity_like


def final_tdd_from_env_info(info_env):
    """Extract the single remaining TDD from ``TensorContractionEnv.env_info()``.

    Returns ``None`` unless the episode fully contracted the network, since a
    partially contracted residual says nothing about equivalence.
    """

    if not info_env.get("fully_contracted"):
        return None
    tdds = info_env.get("tdds") or {}
    if len(tdds) != 1:
        return None
    entry = next(iter(tdds.values()))
    # Entries are [tdd, tensor_count, gate_names].
    return entry[0] if isinstance(entry, (list, tuple)) else entry


def check_final_tdd(
    info_env,
    num_qubits: int | None = None,
    atol: float = 1e-6,
    dense_max_qubits: int = DEFAULT_DENSE_MAX_QUBITS,
) -> dict:
    """Classify the final TDD of a contraction.

    Runs the TDD-native check, plus a dense cross-check for small instances.
    ``status`` is ``identity``, ``not_identity``, ``incomplete`` (the network
    was not fully contracted), ``label_mismatch`` (open legs are not one x/y
    pair per qubit) or ``unavailable``.
    """

    result = {
        "status": STATUS_UNAVAILABLE,
        "is_identity": None,
        "fidelity_like": None,
        "dense_is_identity": None,
        "dense_fidelity_like": None,
        "dense_checked": False,
        "dense_agrees": None,
        "num_open_legs": None,
    }

    if not info_env.get("fully_contracted"):
        result["status"] = STATUS_INCOMPLETE
        return result

    tdd = final_tdd_from_env_info(info_env)
    if tdd is None:
        return result

    labels = tdd_axis_labels(tdd)
    result["num_open_legs"] = len(labels)

    if not _has_full_xy_legs(labels):
        result["status"] = STATUS_LABEL_MISMATCH
        return result

    is_identity, fidelity = check_identity(tdd, atol=atol)
    result["is_identity"] = is_identity
    result["fidelity_like"] = fidelity
    result["status"] = STATUS_IDENTITY if is_identity else STATUS_NOT_IDENTITY

    n = num_qubits if num_qubits is not None else len(qubits_from_labels(labels))
    if n <= dense_max_qubits:
        dense_is_identity, dense_fidelity = check_identity_dense(tdd, atol=atol)
        result["dense_checked"] = True
        result["dense_is_identity"] = dense_is_identity
        result["dense_fidelity_like"] = dense_fidelity
        result["dense_agrees"] = bool(dense_is_identity == is_identity)

    return result
