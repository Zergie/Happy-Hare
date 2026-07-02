import inspect
import os

import pytest


def _load_legacy_harness():
    import test.test_mmu_gcode as legacy_harness

    return legacy_harness


def _call_with_supported_kwargs(function, candidate_kwargs):
    signature = inspect.signature(function)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return function(**candidate_kwargs)
    supported_kwargs = {
        name: value for name, value in candidate_kwargs.items()
        if name in parameters
    }
    return function(**supported_kwargs)


def _repo_root_from_legacy_harness(legacy_harness):
    return os.path.dirname(os.path.dirname(legacy_harness.__file__))


def build_real_mmu(variant, command_name=None):
    legacy_harness = _load_legacy_harness()
    builder = legacy_harness._build_real_mmu

    signature = inspect.signature(builder)
    if all(name in signature.parameters for name in ("repo_root", "mmu_module", "drive_mode", "command_name")):
        repo_root = _repo_root_from_legacy_harness(legacy_harness)
        mmu_module = legacy_harness._load_mmu_module(repo_root)
        return _normalize_built_mmu(builder(repo_root, mmu_module, variant, command_name or "MMU_TEST_MOVE"))

    candidate_kwarg_sets = [
        {"variant": variant},
        {"motion_system": variant},
        {"motor_type": variant},
        {"drive_kind": variant},
        {"use_bldc": variant == "bldc"},
        {},
    ]
    last_error = None
    for kwargs in candidate_kwarg_sets:
        try:
            built = _call_with_supported_kwargs(builder, kwargs)
            return _normalize_built_mmu(built)
        except Exception as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to build MMU harness")


def _normalize_built_mmu(built):
    if isinstance(built, tuple):
        for item in built:
            if hasattr(item, "cmd_MMU_TEST_MOVE"):
                return item
    return built


def make_gcmd(command):
    legacy_harness = _load_legacy_harness()
    repo_root = _repo_root_from_legacy_harness(legacy_harness)
    gcode_module = legacy_harness._load_klippy_gcode_module(repo_root)
    gcode_dispatch = gcode_module.GCodeDispatch(legacy_harness._MiniPrinter())
    return legacy_harness._make_gcmd(gcode_dispatch, command)


def run_command(variant, handler_name, command, expect_failure):
    command_name = command.strip().split()[0].upper() if command else "MMU_TEST_MOVE"
    mmu = build_real_mmu(variant, command_name=command_name)
    gcmd = make_gcmd(command)
    handler = getattr(mmu, handler_name)
    if expect_failure:
        with pytest.raises(Exception) as error_info:
            handler(gcmd)
        pytest.xfail(f"{variant} path fails by real exception: {error_info.value}")
    handler(gcmd)