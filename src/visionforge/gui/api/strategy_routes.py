"""The strategy endpoints: compare, sweep, K-fold, replicates (#204).

Per built-in task, ``POST /api/<task>/{compare,sweep,cv,replicates,
replicated-comparison}``, and a researcher's task's sweep, replicates and
paired comparison under ``/api/custom/{key}/``. Each endpoint builds the task's
runner and hands the request to ``strategies.py``, which validates it, queues
the job and runs it.

``routes.py`` includes the two routers where these routes used to be defined
(the custom-task routers sit between them), so the OpenAPI document and the
route order do not change; it owns the profile lookup and passes it in. The
starters and the K-fold fronts are looked up in ``strategies`` per request, so
a test patches them there.

No ``from __future__ import annotations``: ``ProfileDep`` is built inside the
builders, and FastAPI resolves string annotations against module globals only.
Each runner is imported when a job is submitted, so importing this module stays
light (GUI-11, #202).
"""

import importlib
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from visionforge.core.task_runner import TaskRunner
from visionforge.gui.api import strategies, submission
from visionforge.gui.api.custom_task_routes import _get_custom_task_or_404
from visionforge.gui.api.http_responses import (
    BAD_REQUEST_TEXT,
    NOT_FOUND_TEXT,
    TAG_ANOMALY,
    TAG_CUSTOM_TASKS,
    TAG_DETECTION,
    TAG_EXPERIMENTS,
    TAG_REGRESSION,
    TAG_SEGMENTATION,
    UNPROCESSABLE_TEXT,
    VALIDATION_ERROR_CONTENT,
)
from visionforge.gui.api.profiles import Profile
from visionforge.gui.api.schemas import (
    ComparisonRequest,
    ReplicatedComparisonRequest,
    ReplicatesRequest,
    RunResponse,
    SweepRequest,
    TaskCvRequest,
)
from visionforge.gui.api.submission import (
    _GPU_UNAVAILABLE_TEXT,
    ProfileResolver,
    load_training_stack,
    profile_dependency,
)
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig

# What a comparison and a K-fold document: a bad dataset, a missing one and a
# broken GPU context. A strategy with a search space or seeds also raises 422
# itself (an override under ``output``, duplicate seeds).
_PLAIN_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": BAD_REQUEST_TEXT},
    404: {"description": NOT_FOUND_TEXT},
    503: {"description": _GPU_UNAVAILABLE_TEXT},
}
_SEARCH_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": BAD_REQUEST_TEXT},
    404: {"description": NOT_FOUND_TEXT},
    422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
    503: {"description": _GPU_UNAVAILABLE_TEXT},
}

# Each built-in task's runner (module, class), imported when a job is submitted.
_RUNNERS: dict[type[Any], tuple[str, str]] = {
    ExperimentConfig: (
        "visionforge.blocks.classification_runner",
        "ClassificationRunner",
    ),
    RegressionConfig: ("visionforge.blocks.regression_runner", "RegressionRunner"),
    SegmentationConfig: (
        "visionforge.blocks.segmentation_runner",
        "SegmentationRunner",
    ),
    DetectionConfig: ("visionforge.blocks.detection_runner", "DetectionRunner"),
    AnomalyConfig: ("visionforge.blocks.anomaly_runner", "AnomalyRunner"),
}

# A starter in ``strategies``: (request, config type, runner, profile) -> job.
_Starter = Callable[[Any, type[Any], TaskRunner, Profile], RunResponse]


async def _start_builtin(
    start: _Starter, req: Any, config_type: type[Any], profile: Profile
) -> RunResponse:
    """Load the training stack off the event loop, then start ``config_type``'s job.

    Detection also loads its trainer (ADR-180); with the stack in
    ``sys.modules`` the runner's import takes milliseconds (#258).
    """
    extra = submission._DETECTION_STACK if config_type is DetectionConfig else ()
    await load_training_stack(*extra)
    module, name = _RUNNERS[config_type]
    runner: TaskRunner = getattr(importlib.import_module(module), name)()
    return start(req, config_type, runner, profile)


async def _start_custom(
    start: _Starter, req: Any, key: str, profile: Profile
) -> RunResponse:
    """The same for the custom task ``key``, validated against its own Config.

    Raises:
        HTTPException: 404 for an unknown key.
    """
    await load_training_stack()
    from visionforge.tasks.runner import CustomTaskRunner

    info = _get_custom_task_or_404(key)
    return start(req, info.spec_cls.Config, CustomTaskRunner(info), profile)


def build_search_router(resolve_profile: ProfileResolver) -> APIRouter:
    """Mount the built-in tasks' compare and sweep; ``routes.py`` owns the lookup."""
    router = APIRouter()
    ProfileDep = Annotated[Profile, Depends(profile_dependency(resolve_profile))]

    @router.post(
        "/regression/compare", responses=_PLAIN_RESPONSES, tags=[TAG_REGRESSION]
    )
    async def compare_regression(
        req: ComparisonRequest, profile: ProfileDep
    ) -> RunResponse:
        """Train N regression backbones on the same dataset and rank them (ADR-044)."""
        start = strategies._start_comparison
        return await _start_builtin(start, req, RegressionConfig, profile)

    @router.post(
        "/segmentation/compare", responses=_PLAIN_RESPONSES, tags=[TAG_SEGMENTATION]
    )
    async def compare_segmentation(
        req: ComparisonRequest, profile: ProfileDep
    ) -> RunResponse:
        """Train N segmentation backbones on the same dataset and rank them (ADR-044)."""
        start = strategies._start_comparison
        return await _start_builtin(start, req, SegmentationConfig, profile)

    @router.post(
        "/regression/sweep", responses=_SEARCH_RESPONSES, tags=[TAG_REGRESSION]
    )
    async def sweep_regression(req: SweepRequest, profile: ProfileDep) -> RunResponse:
        """Run a grid/random hyperparameter sweep for regression (ADR-045)."""
        start = strategies._start_sweep
        return await _start_builtin(start, req, RegressionConfig, profile)

    @router.post(
        "/segmentation/sweep", responses=_SEARCH_RESPONSES, tags=[TAG_SEGMENTATION]
    )
    async def sweep_segmentation(req: SweepRequest, profile: ProfileDep) -> RunResponse:
        """Run a grid/random hyperparameter sweep for segmentation (ADR-045)."""
        start = strategies._start_sweep
        return await _start_builtin(start, req, SegmentationConfig, profile)

    @router.post("/detection/compare", responses=_PLAIN_RESPONSES, tags=[TAG_DETECTION])
    async def compare_detection(
        req: ComparisonRequest, profile: ProfileDep
    ) -> RunResponse:
        """Train N detectors on the same dataset and rank them (ADR-044)."""
        start = strategies._start_comparison
        return await _start_builtin(start, req, DetectionConfig, profile)

    @router.post("/detection/sweep", responses=_SEARCH_RESPONSES, tags=[TAG_DETECTION])
    async def sweep_detection(req: SweepRequest, profile: ProfileDep) -> RunResponse:
        """Run a grid/random hyperparameter sweep for detection (ADR-045)."""
        start = strategies._start_sweep
        return await _start_builtin(start, req, DetectionConfig, profile)

    @router.post("/anomaly/compare", responses=_PLAIN_RESPONSES, tags=[TAG_ANOMALY])
    async def compare_anomaly(
        req: ComparisonRequest, profile: ProfileDep
    ) -> RunResponse:
        """Train N anomaly models on the same dataset and rank them (ADR-044)."""
        start = strategies._start_comparison
        return await _start_builtin(start, req, AnomalyConfig, profile)

    @router.post("/anomaly/sweep", responses=_SEARCH_RESPONSES, tags=[TAG_ANOMALY])
    async def sweep_anomaly(req: SweepRequest, profile: ProfileDep) -> RunResponse:
        """Run a grid/random hyperparameter sweep for anomaly detection (ADR-045)."""
        start = strategies._start_sweep
        return await _start_builtin(start, req, AnomalyConfig, profile)

    return router


def build_group_router(resolve_profile: ProfileResolver) -> APIRouter:
    """Mount the remaining strategies; ``routes.py`` owns the profile lookup."""
    router = APIRouter()
    ProfileDep = Annotated[Profile, Depends(profile_dependency(resolve_profile))]

    @router.post(
        "/custom/{key}/sweep", responses=_SEARCH_RESPONSES, tags=[TAG_CUSTOM_TASKS]
    )
    async def sweep_custom_task(
        key: str, req: SweepRequest, profile: ProfileDep
    ) -> RunResponse:
        """Run a grid/random hyperparameter sweep for a custom task (ADR-045/058).

        ``CustomTaskRunner`` gives the task the same orchestrator the built-ins
        use; the search space is validated against the task's own Config. There is
        deliberately no /custom compare endpoint: comparison overrides
        ``model.name``, which ``BaseTaskConfig`` does not guarantee — a one-axis
        sweep over whichever field the task declares covers that use case.
        """
        return await _start_custom(strategies._start_sweep, req, key, profile)

    @router.post(
        "/custom/{key}/replicates",
        responses=_SEARCH_RESPONSES,
        tags=[TAG_CUSTOM_TASKS],
    )
    async def replicates_custom_task(
        key: str, req: ReplicatesRequest, profile: ProfileDep
    ) -> RunResponse:
        """Train one custom-task config N times under different seeds (ADR-056/058)."""
        return await _start_custom(strategies._start_replicates, req, key, profile)

    @router.post("/regression/cv", responses=_PLAIN_RESPONSES, tags=[TAG_REGRESSION])
    async def cv_regression(req: TaskCvRequest, profile: ProfileDep) -> RunResponse:
        """Run K-fold cross-validation for regression (ADR-050) in the background."""
        cv_fn = strategies.run_regression_cross_validation
        return strategies._start_task_cv(
            req, RegressionConfig, cv_fn, "Regression", profile
        )

    @router.post(
        "/segmentation/cv", responses=_PLAIN_RESPONSES, tags=[TAG_SEGMENTATION]
    )
    async def cv_segmentation(req: TaskCvRequest, profile: ProfileDep) -> RunResponse:
        """Run K-fold cross-validation for segmentation in the background."""
        cv_fn = strategies.run_segmentation_cross_validation
        return strategies._start_task_cv(
            req, SegmentationConfig, cv_fn, "Segmentation", profile
        )

    _mount_replicates(router, resolve_profile)
    _mount_replicated_comparisons(router, resolve_profile)
    return router


def _mount_replicates(router: APIRouter, resolve_profile: ProfileResolver) -> None:
    """Mount ``POST /api/<task>/replicates`` for each built-in task."""
    ProfileDep = Annotated[Profile, Depends(profile_dependency(resolve_profile))]

    @router.post(
        "/classification/replicates",
        responses=_SEARCH_RESPONSES,
        tags=[TAG_EXPERIMENTS],
    )
    async def replicates_classification(
        req: ReplicatesRequest, profile: ProfileDep
    ) -> RunResponse:
        """Train one classification config N times under different seeds (ADR-056)."""
        start = strategies._start_replicates
        return await _start_builtin(start, req, ExperimentConfig, profile)

    @router.post(
        "/regression/replicates", responses=_SEARCH_RESPONSES, tags=[TAG_REGRESSION]
    )
    async def replicates_regression(
        req: ReplicatesRequest, profile: ProfileDep
    ) -> RunResponse:
        """Train one regression config N times under different seeds (ADR-056)."""
        start = strategies._start_replicates
        return await _start_builtin(start, req, RegressionConfig, profile)

    @router.post(
        "/segmentation/replicates",
        responses=_SEARCH_RESPONSES,
        tags=[TAG_SEGMENTATION],
    )
    async def replicates_segmentation(
        req: ReplicatesRequest, profile: ProfileDep
    ) -> RunResponse:
        """Train one segmentation config N times under different seeds (ADR-056)."""
        start = strategies._start_replicates
        return await _start_builtin(start, req, SegmentationConfig, profile)

    @router.post(
        "/detection/replicates", responses=_SEARCH_RESPONSES, tags=[TAG_DETECTION]
    )
    async def replicates_detection(
        req: ReplicatesRequest, profile: ProfileDep
    ) -> RunResponse:
        """Train one detection config N times under different seeds (ADR-056)."""
        start = strategies._start_replicates
        return await _start_builtin(start, req, DetectionConfig, profile)

    @router.post("/anomaly/replicates", responses=_SEARCH_RESPONSES, tags=[TAG_ANOMALY])
    async def replicates_anomaly(
        req: ReplicatesRequest, profile: ProfileDep
    ) -> RunResponse:
        """Train one anomaly config N times under different seeds (ADR-056)."""
        start = strategies._start_replicates
        return await _start_builtin(start, req, AnomalyConfig, profile)


def _mount_replicated_comparisons(
    router: APIRouter, resolve_profile: ProfileResolver
) -> None:
    """Mount ``POST /api/<task>/replicated-comparison``, built-ins then custom."""
    ProfileDep = Annotated[Profile, Depends(profile_dependency(resolve_profile))]

    @router.post(
        "/classification/replicated-comparison",
        responses=_SEARCH_RESPONSES,
        tags=[TAG_EXPERIMENTS],
    )
    async def replicated_comparison_classification(
        req: ReplicatedComparisonRequest, profile: ProfileDep
    ) -> RunResponse:
        """Compare classification variants over shared seeds with a paired test."""
        start = strategies._start_replicated_comparison
        return await _start_builtin(start, req, ExperimentConfig, profile)

    @router.post(
        "/regression/replicated-comparison",
        responses=_SEARCH_RESPONSES,
        tags=[TAG_REGRESSION],
    )
    async def replicated_comparison_regression(
        req: ReplicatedComparisonRequest, profile: ProfileDep
    ) -> RunResponse:
        """Compare regression variants over shared seeds with a paired test."""
        start = strategies._start_replicated_comparison
        return await _start_builtin(start, req, RegressionConfig, profile)

    @router.post(
        "/segmentation/replicated-comparison",
        responses=_SEARCH_RESPONSES,
        tags=[TAG_SEGMENTATION],
    )
    async def replicated_comparison_segmentation(
        req: ReplicatedComparisonRequest, profile: ProfileDep
    ) -> RunResponse:
        """Compare segmentation variants over shared seeds with a paired test."""
        start = strategies._start_replicated_comparison
        return await _start_builtin(start, req, SegmentationConfig, profile)

    @router.post(
        "/detection/replicated-comparison",
        responses=_SEARCH_RESPONSES,
        tags=[TAG_DETECTION],
    )
    async def replicated_comparison_detection(
        req: ReplicatedComparisonRequest, profile: ProfileDep
    ) -> RunResponse:
        """Compare detection variants over shared seeds with a paired test."""
        start = strategies._start_replicated_comparison
        return await _start_builtin(start, req, DetectionConfig, profile)

    @router.post(
        "/anomaly/replicated-comparison",
        responses=_SEARCH_RESPONSES,
        tags=[TAG_ANOMALY],
    )
    async def replicated_comparison_anomaly(
        req: ReplicatedComparisonRequest, profile: ProfileDep
    ) -> RunResponse:
        """Compare anomaly variants over shared seeds with a paired test."""
        start = strategies._start_replicated_comparison
        return await _start_builtin(start, req, AnomalyConfig, profile)

    @router.post(
        "/custom/{key}/replicated-comparison",
        responses=_SEARCH_RESPONSES,
        tags=[TAG_CUSTOM_TASKS],
    )
    async def replicated_comparison_custom(
        key: str, req: ReplicatedComparisonRequest, profile: ProfileDep
    ) -> RunResponse:
        """Compare a custom task's variants over shared seeds (ADR-058 + ADR-061)."""
        start = strategies._start_replicated_comparison
        return await _start_custom(start, req, key, profile)
