"""GFN-xTB engine, driving the `xtb` command-line program.

Parsing is written against real xtb 6.7.1 output. Three details that bite:

* xtb prints several ``SUMMARY`` blocks during an optimisation; only the last
  one holds the converged energy.
* ``normal termination of xtb`` goes to **stderr**, not stdout. A parser that
  only reads stdout will call every successful run a failure.
* ``--uhf`` takes the number of *unpaired electrons*, not the multiplicity.
  Passing multiplicity there silently computes the wrong spin state, which is
  precisely the failure mode that makes radical BDEs look plausible and wrong.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from ..calc import (
    CalculationError,
    CalculationResult,
    CalculationSpec,
    ResourceUsage,
    Task,
    geometry_hash,
)
from ..fidelity import Fidelity, Tier
from ..geometry import Conformer, ConformerEnsemble
from ..units import HARTREE, Quantity, QuantityKind
from .base import EngineError

__all__ = ["XTBEngine", "parse_g98_modes"]


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value

def parse_g98_modes(text: str) -> list[tuple[float, np.ndarray]]:
    """Parse (frequency cm^-1, displacement vectors) from xtb's g98.out.

    Modes come in column blocks of up to three. Used to escape saddle points by
    displacing along an imaginary mode.
    """
    lines = text.splitlines()
    modes: list[tuple[float, list[list[float]]]] = []
    i = 0
    while i < len(lines):
        if "Frequencies --" not in lines[i]:
            i += 1
            continue
        freqs = [float(v) for v in lines[i].split("--")[1].split()]
        j = i + 1
        while j < len(lines) and "Atom AN" not in lines[j]:
            j += 1
        j += 1
        blocks: list[list[list[float]]] = [[] for _ in freqs]
        while j < len(lines):
            parts = lines[j].split()
            if len(parts) < 2 + 3 * len(freqs):
                break
            values = [float(v) for v in parts[2 : 2 + 3 * len(freqs)]]
            for k in range(len(freqs)):
                blocks[k].append(values[3 * k : 3 * k + 3])
            j += 1
        for freq, disp in zip(freqs, blocks):
            modes.append((freq, disp))
        i = j
    return [(f, np.array(d, dtype=float)) for f, d in modes]


#: GFN methods this engine knows, mapped to the value of xtb's --gfn flag.
_GFN_METHODS = {
    "gfn2-xtb": "2",
    "gfn1-xtb": "1",
    "gfn0-xtb": "0",
}
#: GFN-FF has its own flag rather than a --gfn level.
_FORCE_FIELD_METHOD = "gfn-ff"

#: xtb's implicit solvation models and the flag that selects them.
_SOLVATION_FLAGS = {"alpb": "--alpb", "gbsa": "--gbsa", "cpcmx": "--cpcmx"}

_TASK_FLAGS = {
    Task.SINGLE_POINT: [],
    Task.OPTIMIZE: ["--opt"],
    Task.OPT_FREQ: ["--ohess"],
    Task.HESSIAN: ["--hess"],
}

_VERSION_RE = re.compile(r"xtb version\s+(\S+)")
_SUMMARY_ENERGY_RE = re.compile(r"::\s*total energy\s+(-?\d+\.\d+)\s*Eh")
_TOTAL_ENERGY_RE = re.compile(r"\|\s*TOTAL ENERGY\s+(-?\d+\.\d+)\s*Eh")
_TOTAL_ENTHALPY_RE = re.compile(r"\|\s*TOTAL ENTHALPY\s+(-?\d+\.\d+)\s*Eh")
_TOTAL_GIBBS_RE = re.compile(r"\|\s*TOTAL FREE ENERGY\s+(-?\d+\.\d+)\s*Eh")
_ZPE_RE = re.compile(r"::\s*zero point energy\s+(-?\d+\.\d+)\s*Eh")
_IMAG_RE = re.compile(r"#\s*imaginary freq\.\s+(\d+)")
_OPT_CONVERGED_RE = re.compile(r"GEOMETRY OPTIMIZATION CONVERGED")
_THERMO_TABLE_RE = re.compile(
    r"T/K\s+H\(0\)-H\(T\)\+PV.*?\n\s*-+\s*\n\s*(\d+\.\d+)", re.S
)
_OPT_FAILED_RE = re.compile(r"FAILED TO CONVERGE GEOMETRY OPTIMIZATION")


class XTBEngine:
    """Runs `xtb` as a subprocess in a scratch directory."""

    name = "xtb"

    def __init__(
        self,
        executable: str | None = None,
        *,
        n_cores: int = 1,
        scratch_root: str | Path | None = None,
        usd_per_core_hour: float | None = None,
        accuracy: float | None = None,
        electronic_temperature_K: float | None = None,
        imaginary_retries: int = 0,
        imaginary_step_angstrom: float = 0.3,
        timeout_seconds: float = 600.0,
    ):
        self.executable = executable or shutil.which("xtb") or "xtb"
        self.n_cores = int(n_cores)
        self.scratch_root = Path(scratch_root) if scratch_root else None
        self.usd_per_core_hour = usd_per_core_hour
        self.accuracy = accuracy
        self.electronic_temperature_K = electronic_temperature_K
        # A screening campaign must not be able to stall on one pathological
        # molecule. Ten minutes is generous for GFN2 on a drug-sized system.
        self.timeout_seconds = float(timeout_seconds)
        self.imaginary_retries = int(imaginary_retries)
        self.imaginary_step_angstrom = float(imaginary_step_angstrom)
        self._version: str | None = None

    # -- availability ------------------------------------------------------

    def is_available(self) -> bool:
        return shutil.which(self.executable) is not None or Path(self.executable).exists()

    def version(self) -> str:
        if self._version is None:
            if not self.is_available():
                raise EngineError(f"xtb executable not found at {self.executable!r}")
            out = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                timeout=60,
                # xtb writes UTF-8 (alpha, cm^-1). Windows would otherwise
                # decode as cp1252 and crash on the first Greek letter.
                encoding="utf-8",
                errors="replace",
            )
            match = _VERSION_RE.search(out.stdout + out.stderr)
            self._version = match.group(1) if match else "unknown"
        return self._version

    def supports(self, fidelity: Fidelity, task: Task) -> bool:
        if fidelity.tier not in (Tier.SEMIEMPIRICAL, Tier.FORCE_FIELD):
            return False
        method = fidelity.method.lower()
        if method not in _GFN_METHODS and method != _FORCE_FIELD_METHOD:
            return False
        if fidelity.basis:
            return False  # a semiempirical method has no basis to choose
        if fidelity.solvation_model and fidelity.solvation_model.lower() not in _SOLVATION_FLAGS:
            return False
        return task in _TASK_FLAGS

    # -- command construction ---------------------------------------------

    def _command(self, xyz: str, fidelity: Fidelity, task: Task, spec: CalculationSpec) -> list[str]:
        method = fidelity.method.lower()
        cmd = [self.executable, xyz]
        if method == _FORCE_FIELD_METHOD:
            cmd.append("--gfnff")
        else:
            cmd += ["--gfn", _GFN_METHODS[method]]
        cmd += ["--chrg", str(spec.charge)]
        # --uhf is the count of UNPAIRED ELECTRONS, not the multiplicity.
        cmd += ["--uhf", str(spec.n_unpaired)]
        cmd += _TASK_FLAGS[task]
        if fidelity.solvation_model:
            flag = _SOLVATION_FLAGS[fidelity.solvation_model.lower()]
            cmd += [flag, fidelity.solvation_solvent]
        if task.yields_thermochemistry:
            # xtb has no --temp flag; the thermostatistical temperature is set
            # through an xcontrol $thermo block. Always written explicitly so
            # the recorded temperature is the one that actually ran.
            cmd += ["--input", "xcontrol"]
        if self.electronic_temperature_K is not None:
            cmd += ["--etemp", str(self.electronic_temperature_K)]
        if self.accuracy is not None:
            cmd += ["--acc", str(self.accuracy)]
        cmd += ["--parallel", str(self.n_cores)]
        return cmd

    # -- parsing -----------------------------------------------------------

    @staticmethod
    def _parse(stdout: str, stderr: str, task: Task) -> dict:
        """Pull energies, imaginary-mode count and convergence out of xtb output."""
        out: dict = {"warnings": []}

        # The final SUMMARY block holds the converged energy; the boxed
        # TOTAL ENERGY line is only printed by thermochemistry runs.
        boxed = _TOTAL_ENERGY_RE.findall(stdout)
        summaries = _SUMMARY_ENERGY_RE.findall(stdout)
        if boxed:
            out["energy_hartree"] = float(boxed[-1])
        elif summaries:
            out["energy_hartree"] = float(summaries[-1])
        else:
            out["energy_hartree"] = None

        enthalpy = _TOTAL_ENTHALPY_RE.findall(stdout)
        gibbs = _TOTAL_GIBBS_RE.findall(stdout)
        zpe = _ZPE_RE.findall(stdout)
        out["enthalpy_hartree"] = float(enthalpy[-1]) if enthalpy else None
        out["gibbs_hartree"] = float(gibbs[-1]) if gibbs else None
        out["zpe_hartree"] = float(zpe[-1]) if zpe else None

        imag = _IMAG_RE.findall(stdout)
        out["n_imaginary"] = int(imag[-1]) if imag else None

        table = _THERMO_TABLE_RE.search(stdout)
        out["thermo_temperature_K"] = float(table.group(1)) if table else None

        # "normal termination of xtb" is written to stderr.
        terminated = "normal termination" in (stderr + stdout)
        if task.changes_geometry:
            if _OPT_FAILED_RE.search(stdout):
                out["converged"] = False
                out["warnings"].append("geometry optimisation failed to converge")
            elif _OPT_CONVERGED_RE.search(stdout):
                out["converged"] = terminated
            else:
                out["converged"] = False
                out["warnings"].append("no optimisation convergence marker in output")
        else:
            out["converged"] = terminated and out["energy_hartree"] is not None

        if task.yields_thermochemistry and out["gibbs_hartree"] is None:
            out["warnings"].append("thermochemistry requested but no free energy parsed")
        return out

    @staticmethod
    def _read_optimised_xyz(workdir: Path, expected_elements: tuple[str, ...]) -> np.ndarray | None:
        path = workdir / "xtbopt.xyz"
        if not path.exists():
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        n = int(lines[0].strip())
        symbols, coords = [], []
        for row in lines[2 : 2 + n]:
            parts = row.split()
            symbols.append(parts[0])
            coords.append([float(v) for v in parts[1:4]])
        if tuple(symbols) != expected_elements:
            raise CalculationError(
                "xtb returned a geometry with a different atom ordering than it was "
                "given; every downstream atom index would be wrong"
            )
        return np.array(coords, dtype=float)

    # -- execution ---------------------------------------------------------

    def run(
        self,
        ensemble: ConformerEnsemble,
        fidelity: Fidelity,
        task: Task,
        *,
        temperature_K: float = 298.15,
        timeout_seconds: float | None = None,
    ) -> list[CalculationResult]:
        if not self.is_available():
            raise EngineError(f"xtb executable not found at {self.executable!r}")
        if not self.supports(fidelity, task):
            raise EngineError(
                f"XTBEngine cannot run {fidelity.label()} / {task.value} exactly; "
                f"fix the fidelity rather than substituting a nearby method"
            )

        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        results: list[CalculationResult] = []
        for index, conformer in enumerate(ensemble):
            spec = CalculationSpec(
                molecule=ensemble.molecule,
                fidelity=fidelity,
                task=task,
                geometry_hash=geometry_hash(conformer.coords, ensemble.elements),
                charge=ensemble.charge,
                multiplicity=ensemble.multiplicity,
                temperature_K=temperature_K,
            )
            results.append(
                self._run_one(ensemble, conformer, index, spec, fidelity, task, timeout)
            )
        return results

    def _run_one(
        self,
        ensemble: ConformerEnsemble,
        conformer: Conformer,
        index: int,
        spec: CalculationSpec,
        fidelity: Fidelity,
        task: Task,
        timeout_seconds: float,
    ) -> CalculationResult:
        parent = str(self.scratch_root) if self.scratch_root else None
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"xtb_{spec.id}_", dir=parent) as tmp:
            workdir = Path(tmp)
            xyz_path = workdir / "input.xyz"
            xyz_path.write_text(ensemble.to_xyz(index), encoding="utf-8")
            if task.yields_thermochemistry and ensemble.n_atoms > 1:
                (workdir / "xcontrol").write_text(
                    "\n".join(["$thermo", f"   temp={spec.temperature_K}", "$end", ""]),
                    encoding="utf-8",
                )

            # A single atom has no vibrations. xtb refuses --ohess for it, and
            # a numerical Hessian would be the wrong tool anyway: the ideal-gas
            # corrections are analytic and exact. Run a single point and add
            # them afterwards. This is the hydrogen atom every X-H homolysis
            # produces, so it is a main path, not an edge case.
            monatomic = ensemble.n_atoms == 1
            effective_task = (
                Task.SINGLE_POINT
                if monatomic and task.yields_thermochemistry
                else task
            )
            cmd = self._command("input.xyz", fidelity, effective_task, spec)
            env = dict(os.environ)
            env.setdefault("OMP_NUM_THREADS", str(self.n_cores))
            env.setdefault("MKL_NUM_THREADS", str(self.n_cores))

            def invoke() -> tuple[str, str, bool, float]:
                started = time.perf_counter()
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=workdir,
                        capture_output=True,
                        text=True,
                        timeout=timeout_seconds,
                        env=env,
                        encoding="utf-8",
                        errors="replace",
                    )
                    return (
                        _as_text(proc.stdout),
                        _as_text(proc.stderr),
                        False,
                        time.perf_counter() - started,
                    )
                except subprocess.TimeoutExpired as exc:
                    return (
                        _as_text(exc.stdout),
                        _as_text(exc.stderr),
                        True,
                        time.perf_counter() - started,
                    )

            stdout, stderr, timed_out, wall = invoke()
            parsed = self._parse(stdout, stderr, effective_task)

            # A structure with imaginary modes is a saddle point, not a minimum,
            # and its thermochemistry is not the free energy of a stable species.
            # Displace along the most negative mode and re-optimise: that fixes
            # torsional saddles outright, and breaks the symmetry of a
            # Jahn-Teller case so the optimiser can find the distorted minimum.
            retries_left = self.imaginary_retries if task.yields_thermochemistry else 0
            retry_notes: list[str] = []
            while (
                retries_left > 0
                and not timed_out
                and parsed.get("n_imaginary")
                and (workdir / "g98.out").exists()
                and (workdir / "xtbopt.xyz").exists()
            ):
                modes = parse_g98_modes(
                    (workdir / "g98.out").read_text(encoding="utf-8", errors="replace")
                )
                imaginary = [m for m in modes if m[0] < 0]
                if not imaginary:
                    break
                freq, displacement = min(imaginary, key=lambda m: m[0])
                current = self._read_optimised_xyz(workdir, ensemble.elements)
                if current is None or displacement.shape != current.shape:
                    break
                displaced = current + self.imaginary_step_angstrom * displacement
                xyz_path.write_text(
                    ConformerEnsemble(
                        molecule=ensemble.molecule,
                        elements=ensemble.elements,
                        conformers=[Conformer(coords=displaced)],
                        protocol=ensemble.protocol,
                        charge=ensemble.charge,
                        multiplicity=ensemble.multiplicity,
                    ).to_xyz(0),
                    encoding="utf-8",
                )
                retry_notes.append(
                    f"displaced {self.imaginary_step_angstrom} A along the "
                    f"{freq:.0f} cm^-1 imaginary mode and re-optimised"
                )
                out2, err2, timed_out, extra = invoke()
                wall += extra
                parsed2 = self._parse(out2, err2, effective_task)
                retries_left -= 1
                improved = (parsed2.get("n_imaginary") or 0) < (
                    parsed.get("n_imaginary") or 0
                )
                if improved or parsed2.get("n_imaginary") == 0:
                    stdout, stderr, parsed = out2, err2, parsed2
                    parsed["warnings"].extend(retry_notes)
                else:
                    parsed["warnings"].extend(retry_notes)
                    parsed["warnings"].append(
                        "mode following did not remove the imaginary frequencies"
                    )
                    break
                if not parsed.get("n_imaginary"):
                    break
            actual_T = parsed.get("thermo_temperature_K")
            if actual_T is not None and abs(actual_T - spec.temperature_K) > 1e-6:
                raise CalculationError(
                    f"xtb reported thermochemistry at {actual_T} K but the spec "
                    f"asks for {spec.temperature_K} K; the free energy would be "
                    f"attributed to the wrong temperature"
                )
            if timed_out:
                parsed["converged"] = False
                parsed["warnings"].append(f"timed out after {timeout_seconds}s")

            phase = spec.phase

            geometry = None
            if effective_task.changes_geometry and parsed["converged"]:
                coords = self._read_optimised_xyz(workdir, ensemble.elements)
                if coords is not None:
                    geometry = Conformer(coords=coords, converged=True, label=conformer.label)

            def q(value, kind):
                if value is None:
                    return None
                temp = None if kind is QuantityKind.ELECTRONIC_ENERGY else spec.temperature_K
                if kind is QuantityKind.ZERO_POINT_ENERGY:
                    temp = None
                return Quantity(value, HARTREE, kind, phase, temp)

            if monatomic and task.yields_thermochemistry and parsed["energy_hartree"] is not None:
                from ..thermo import monatomic_thermochemistry

                e = Quantity(
                    parsed["energy_hartree"], HARTREE,
                    QuantityKind.ELECTRONIC_ENERGY, phase, None,
                )
                h, g = monatomic_thermochemistry(
                    e,
                    ensemble.elements[0],
                    temperature_K=spec.temperature_K,
                    multiplicity=ensemble.multiplicity,
                )
                parsed["enthalpy_hartree"] = h.to(HARTREE).value
                parsed["gibbs_hartree"] = g.to(HARTREE).value
                parsed["zpe_hartree"] = 0.0
                parsed["n_imaginary"] = 0
                parsed["warnings"].append(
                    "monatomic species: single point plus analytic ideal-gas "
                    "thermochemistry (Sackur-Tetrode), not a numerical Hessian"
                )

            return CalculationResult(
                spec=spec,
                converged=bool(parsed["converged"]),
                usage=ResourceUsage(
                    wall_seconds=wall,
                    n_cores=self.n_cores,
                    usd_per_core_hour=self.usd_per_core_hour,
                ),
                engine=self.name,
                engine_version=self.version(),
                energy=q(parsed["energy_hartree"], QuantityKind.ELECTRONIC_ENERGY),
                enthalpy=q(parsed["enthalpy_hartree"], QuantityKind.ENTHALPY),
                gibbs=q(parsed["gibbs_hartree"], QuantityKind.GIBBS),
                zpe=q(parsed["zpe_hartree"], QuantityKind.ZERO_POINT_ENERGY),
                geometry=geometry,
                n_imaginary=parsed["n_imaginary"],
                warnings=tuple(parsed["warnings"]),
                raw_output=stdout,
                metadata={"command": " ".join(cmd), "conformer_index": index},
            )
