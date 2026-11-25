"""Generate all LOKI corpora: python -m loki.collect_all --help."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from .collectors import COLLECTORS


REPO = Path(__file__).resolve().parents[1]

# Each key is both the corpus directory name and the preset used to construct
# its collector. The remaining presets have the same stored representation,
# proposal distribution, and validity rules, so they consume that corpus.
CORPUS_GROUPS = {
    "BallCatcher-Pitch": (
        "BallCatcher-Pitch", "BallCatcher-Arc", "BallCatcher-Fielding"),
    "HalfCheetah-v4": ("HalfCheetah-v4",),
    "Swimmer-v4": ("Swimmer-v4",),
    "Hopper-v4": ("Hopper-v4",),
    "Walker2d-v4": ("Walker2d-v4",),
    "Ant-v4": ("Ant-v4",),
    "Humanoid-v4": ("Humanoid-v4", "HumanoidStandup-v4"),
    "Microgrid-Random": (
        "Microgrid-Random", "Microgrid-Forecast", "Microgrid-OffGridSingle"),
    "Microgrid-Campus": ("Microgrid-Campus", "Microgrid-OffGridCampus"),
    "Racing-Spielberg": ("Racing-Spielberg", "Racing-Levine"),
    "SolarCleaner-Reach": (
        "SolarCleaner-Reach", "SolarCleaner-Stability", "SolarCleaner-Traverse"),
    "TruckUnload-Single": ("TruckUnload-Single",),
    "SoftWalker-Beam3D": ("SoftWalker-Beam3D",),
    # RotateInHand deforms the same hand as HandPickUp: an identical design space and validity rules.
    "HandPickUp-Cube": (
        "HandPickUp-Cube", "HandPickUp-SquashBall", "HandPickUp-Hammer", "RotateInHand-Cube"),
    "NeroGrasp-All": tuple(sorted(
        preset for preset in COLLECTORS if preset.startswith("NeroGrasp"))),
    **{
        f"Pokenv-LeagueG{generation}": (
            f"Pokenv-LeagueG{generation}", f"Pokenv-LeagueG{generation}H")
        for generation in range(1, 9)
    },
    "Pokenv-LeagueG9": (
        "Pokenv-LeagueG9", "Pokenv-LeagueG9H",
        "Pokenv-LeagueRand", "Pokenv-LeagueRandH"),
    "Pokenv-Random": ("Pokenv-Random",),
    "Pokenv-MaxPower": ("Pokenv-MaxPower",),
    "Pokenv-Heuristic": ("Pokenv-Heuristic",),
    "LQStructure-LQR": ("LQStructure-LQR",),
    "LQStructure-LQG": ("LQStructure-LQG",),
    "LQStructure-QP": ("LQStructure-QP",),
    "Network-Urban": ("Network-Urban",),
    "Network-Suburban": ("Network-Suburban",),
    "Network-Motorway": ("Network-Motorway",),
    "Warehouse-Small": ("Warehouse-Small",),
    "Warehouse-Medium": ("Warehouse-Medium",),
    "Warehouse-Congested": ("Warehouse-Congested",),
}
# Corpus directories (and the VAE/cluster artifacts under artifacts/loki) keep the ids they were collected under
# before codesign-gym renamed its env ids (378efcd): their manifests and artifacts record those paths and pin the
# manifests by hash, so they are not renamed. Every corpus/artifact path goes through corpus_dir().
CORPUS_DIRS = {"LQStructure-LQR": "LQStructure-LQR-v0", "LQStructure-LQG": "LQStructure-LQG-v0",
    "LQStructure-QP": "LQStructure-QP-v0", "Microgrid-Random": "MicrogridRandom",
    "Microgrid-Forecast": "MicrogridForecast", "Microgrid-Campus": "MicrogridCampus",
    "Microgrid-OffGridSingle": "MicrogridOffGridSingle", "Microgrid-OffGridCampus": "MicrogridOffGridCampus",
    "Pokenv-Random": "PokenvRandom", "Pokenv-MaxPower": "PokenvMaxPower", "Pokenv-Heuristic": "PokenvHeuristic",
    "Network-Urban": "NetworkUrban", "Network-Suburban": "NetworkSuburban", "Network-Motorway": "NetworkMotorway",
    "Warehouse-Small": "WarehouseSmall", "Warehouse-Medium": "WarehouseMedium",
    "Warehouse-Congested": "WarehouseCongested", "Racing-Spielberg": "RacingSpielberg",
    "Racing-Levine": "RacingLevine", "SolarCleaner-Reach": "SolarCleanerReach",
    "SolarCleaner-Stability": "SolarCleanerStability", "SolarCleaner-Traverse": "SolarCleanerTraverse",
    "RotateInHand-Cube": "RotateInHand", "TruckUnload-Single": "TruckUnloadSingle",
    "TruckUnload-Multi": "TruckUnloadMulti", "SoftWalker-Spring2D": "SoftWalkerSpring2D",
    "SoftWalker-MPM2D": "SoftWalkerMPM2D", "SoftWalker-MPM3D": "SoftWalkerMPM3D",
    "SoftWalker-Beam3D": "SoftWalkerBeam3D", "SoftWalker-HexFEM3D": "SoftWalkerHexFEM3D",
    "SoftWalker-Spring2DStairs": "SoftWalkerSpring2D-Stairs", "SoftWalker-Spring2DGaps": "SoftWalkerSpring2D-Gaps",
    "SoftWalker-MPM2DStairs": "SoftWalkerMPM2D-Stairs", "SoftWalker-MPM2DGaps": "SoftWalkerMPM2D-Gaps",
    "SoftWalker-MPM3DStairs": "SoftWalkerMPM3D-Stairs", "SoftWalker-MPM3DGaps": "SoftWalkerMPM3D-Gaps",
    "SoftWalker-Beam3DStairs": "SoftWalkerBeam3D-Stairs", "SoftWalker-Beam3DGaps": "SoftWalkerBeam3D-Gaps",
    "SoftWalker-HexFEM3DStairs": "SoftWalkerHexFEM3D-Stairs", "SoftWalker-HexFEM3DGaps": "SoftWalkerHexFEM3D-Gaps",
    "NeroGrasp-All": "NeroGraspAll", "NeroGrasp-Small": "NeroGraspSmall", "NeroGrasp-Tiny": "NeroGraspTiny",
    "NeroGrasp-Thin": "NeroGraspThin", "NeroGrasp-SetScrew": "NeroGraspSetScrew", "NeroGrasp-TNut": "NeroGraspTNut",
    "NeroGrasp-LockNut": "NeroGraspLockNut", "NeroGrasp-Screw": "NeroGraspScrew",
    "NeroGrasp-Scissors": "NeroGraspScissors", "NeroGrasp-Tweezers": "NeroGraspTweezers",
    "NeroGrasp-Spatula": "NeroGraspSpatula", "NeroGrasp-Stick": "NeroGraspStick",
    "NeroGrasp-ScrewLong": "NeroGraspScrewLong", "Pokenv-LeagueG1": "PokenvLeague-G1",
    "Pokenv-LeagueG1H": "PokenvLeague-G1H", "Pokenv-LeagueG2": "PokenvLeague-G2",
    "Pokenv-LeagueG2H": "PokenvLeague-G2H", "Pokenv-LeagueG3": "PokenvLeague-G3",
    "Pokenv-LeagueG3H": "PokenvLeague-G3H", "Pokenv-LeagueG4": "PokenvLeague-G4",
    "Pokenv-LeagueG4H": "PokenvLeague-G4H", "Pokenv-LeagueG5": "PokenvLeague-G5",
    "Pokenv-LeagueG5H": "PokenvLeague-G5H", "Pokenv-LeagueG6": "PokenvLeague-G6",
    "Pokenv-LeagueG6H": "PokenvLeague-G6H", "Pokenv-LeagueG7": "PokenvLeague-G7",
    "Pokenv-LeagueG7H": "PokenvLeague-G7H", "Pokenv-LeagueG8": "PokenvLeague-G8",
    "Pokenv-LeagueG8H": "PokenvLeague-G8H", "Pokenv-LeagueG9": "PokenvLeague-G9",
    "Pokenv-LeagueG9H": "PokenvLeague-G9H", "Pokenv-LeagueRand": "PokenvLeague-Rand",
    "Pokenv-LeagueRandH": "PokenvLeague-RandH"}


def corpus_dir(corpus_id):
    """Directory name of a corpus under data/loki (and of its artifacts under artifacts/loki)."""
    return CORPUS_DIRS.get(corpus_id, corpus_id)


PRESET_TO_CORPUS = {
    preset: corpus for corpus, presets in CORPUS_GROUPS.items() for preset in presets
}
_GROUPED_PRESETS = [preset for presets in CORPUS_GROUPS.values() for preset in presets]
if (len(_GROUPED_PRESETS) != len(set(_GROUPED_PRESETS))
        or set(_GROUPED_PRESETS) != set(COLLECTORS)):
    raise RuntimeError("CORPUS_GROUPS must contain every collector preset exactly once")
if any(not presets or presets[0] != corpus for corpus, presets in CORPUS_GROUPS.items()):
    raise RuntimeError("Each canonical corpus ID must be the group canonical preset")


def selected_corpora(presets):
    """Map an optional preset selection to unique canonical corpus IDs."""
    if presets is None:
        return list(CORPUS_GROUPS)
    return list(dict.fromkeys(PRESET_TO_CORPUS[preset] for preset in presets))


def grouped_overrides(overrides, corpora, parser):
    """Resolve overrides supplied under any member of a shared group."""
    resolved = {}
    for corpus in corpora:
        supplied = [(preset, overrides[preset]) for preset in CORPUS_GROUPS[corpus]
                    if preset in overrides]
        if not supplied:
            resolved[corpus] = {}
            continue
        value = supplied[0][1]
        if any(candidate != value for _, candidate in supplied[1:]):
            names = ", ".join(name for name, _ in supplied)
            parser.error(f"conflicting overrides for shared corpus {corpus}: {names}")
        resolved[corpus] = value
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "data/loki")
    parser.add_argument("--count", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cuda:0",
                        help="Use CUDA_VISIBLE_DEVICES to choose a physical GPU")
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--audit-every", type=int, default=10_000)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--max-stalled-attempts", type=int, default=10_000)
    parser.add_argument("--presets", nargs="+", choices=sorted(COLLECTORS),
                        help="Optional preset subset; members of one corpus group are deduplicated")
    parser.add_argument("--pokemon-source", type=Path,
                        help="Pokémon source checkout; defaults to sibling pokemon-codesign-env if present")
    parser.add_argument("--overrides", type=Path,
                        help='JSON mapping of preset to native kwargs, e.g. {"NeroGrasp-Tiny": {"sample_perturbation_m": 0.001}}')
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.count, args.shard_size, args.max_stalled_attempts) < 1:
        parser.error("count, shard-size, and max-stalled-attempts must be positive")
    if min(args.seed, args.audit_every) < 0 or (args.max_attempts is not None and args.max_attempts < 1):
        parser.error("seed/audit-every must be nonnegative; max-attempts must be positive")
    overrides = {}
    if args.overrides:
        try:
            overrides = json.loads(args.overrides.read_text())
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if (not isinstance(overrides, dict) or set(overrides) - COLLECTORS.keys()
                or not all(isinstance(value, dict) for value in overrides.values())):
            parser.error("overrides must map supported preset IDs to JSON objects")
    source = args.pokemon_source
    if source is None and (REPO.parent / "pokemon-codesign-env/src/pokemon_codesign_env").is_dir():
        source = REPO.parent / "pokemon-codesign-env"
    if source is not None:
        source = source.resolve()
        if not (source / "src/pokemon_codesign_env").is_dir():
            parser.error("pokemon-source must contain src/pokemon_codesign_env")
    root = args.output_dir.resolve()
    requested_presets = list(dict.fromkeys(args.presets or sorted(COLLECTORS)))
    corpora = selected_corpora(args.presets)
    overrides_by_corpus = grouped_overrides(overrides, corpora, parser)
    results = []
    started = time.monotonic()
    for index, corpus_id in enumerate(corpora, 1):
        served_presets = CORPUS_GROUPS[corpus_id]
        corpus = root / corpus_dir(corpus_id)
        command = [sys.executable, "-u", "-m", "loki.collectors", "--env-id", corpus_id,
                   "--output-dir", str(corpus), "--count", str(args.count),
                   "--seed", str(args.seed), "--device", args.device,
                   "--shard-size", str(args.shard_size), "--audit-every", str(args.audit_every),
                   "--max-stalled-attempts", str(args.max_stalled_attempts),
                   "--env-kwargs", json.dumps(overrides_by_corpus[corpus_id])]
        if args.max_attempts is not None:
            command += ["--max-attempts", str(args.max_attempts)]
        if (corpus / "manifest.json").exists():
            command.append("--resume")
        env = os.environ.copy()
        # Avoid JAX reserving most GPU memory during Warehouse/Racing startup.
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        if corpus_id.startswith(("Racing", "Warehouse")):
            # CUDA ordinal zero is relative to inherited CUDA_VISIBLE_DEVICES.
            env["JAX_PLATFORMS"] = "cpu" if args.device == "cpu" else "cuda"
            env["JAX_CUDA_VISIBLE_DEVICES"] = "0"
        if corpus_id.startswith("Pokenv") and source is not None:
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, [str(source / "src"), env.get("PYTHONPATH")]))
        served = ", ".join(served_presets)
        print(f"[{index}/{len(corpora)}] {corpus_id} serves {served}", flush=True)
        print(f"  {shlex.join(command)}", flush=True)
        if args.dry_run:
            if corpus_id.startswith("Pokenv") and source is not None:
                print(f"  Pokémon source: {source}", flush=True)
            continue
        logs = root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{corpus_dir(corpus_id)}.log"
        print(f"  Log: {log_path}", flush=True)
        run_started = time.monotonic()
        with log_path.open("a") as log:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log.write(f"\nLaunch {stamp}: {shlex.join(command)}\n")
            log.flush()
            process = subprocess.Popen(
                command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait()
            except KeyboardInterrupt:
                # Terminal SIGINT also reaches the collector, which flushes its
                # shard. Explicitly signal it for interrupts sent only to us.
                import signal
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    process.kill()
                    process.wait()
                code = 130
        result = {"corpus_id": corpus_id, "env_id": corpus_id,
                  "serves_presets": list(served_presets), "returncode": code,
                  "elapsed_seconds": time.monotonic() - run_started,
                  "log": str(log_path)}
        results.append(result)
        summary = {"target_per_corpus": args.count,
                   "requested_presets": requested_presets,
                   "requested_corpora": corpora,
                   "results": results,
                   "elapsed_seconds": time.monotonic() - started}
        summary_path = root / "collection_summary.json"
        temporary = root / "collection_summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(summary_path)
        label = "Complete" if code == 0 else "Failed/interrupted"
        print(f"  {label} (exit {code})", flush=True)
        if code in (130, -2):
            return 130
    if args.dry_run:
        return 0
    failed = [result["corpus_id"] for result in results if result["returncode"] != 0]
    summary_path = root / "collection_summary.json"
    print(f"Finished: {len(results) - len(failed)}/{len(corpora)} complete. "
          f"Summary: {summary_path}", flush=True)
    if failed:
        print("Failed/incomplete corpora: " + ", ".join(failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
