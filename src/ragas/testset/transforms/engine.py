from __future__ import annotations

import asyncio
import logging
import typing as t
import concurrent.futures
from ragas.executor import is_event_loop_running
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.transforms.base import BaseGraphTransformation

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)

Transforms = t.Union[
    t.List[BaseGraphTransformation],
    "Parallel",
    BaseGraphTransformation,
]


class Parallel:
    """
    Collection of transformations to be applied in parallel.

    Examples
    --------
    >>> Parallel(HeadlinesExtractor(), SummaryExtractor())
    """

    def __init__(self, *transformations: BaseGraphTransformation):
        self.transformations = list(transformations)

    def generate_execution_plan(self, kg: KnowledgeGraph) -> t.List[t.Coroutine]:
        coroutines = []
        for transformation in self.transformations:
            coroutines.extend(transformation.generate_execution_plan(kg))
        return coroutines


async def run_coroutines(coroutines: t.List[t.Coroutine], desc: str, max_workers: int):
    """
    Run a list of coroutines in parallel using gather.
    """
    # Create tasks for all coroutines
    tasks = [asyncio.create_task(coro) for coro in coroutines]

    # Use gather to run all tasks in parallel
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Error during transformation: {e}")


def get_desc(transform: BaseGraphTransformation | Parallel):
    if isinstance(transform, Parallel):
        transform_names = [t.__class__.__name__ for t in transform.transformations]
        return f"Applying [{', '.join(transform_names)}]"
    else:
        return f"Applying {transform.__class__.__name__}"


def apply_nest_asyncio():
    NEST_ASYNCIO_APPLIED: bool = False
    if is_event_loop_running():
        # an event loop is running so call nested_asyncio to fix this
        try:
            import nest_asyncio
        except ImportError:
            raise ImportError(
                "It seems like your running this in a jupyter-like environment. Please install nest_asyncio with `pip install nest_asyncio` to make it work."
            )

        if not NEST_ASYNCIO_APPLIED:
            nest_asyncio.apply()
            NEST_ASYNCIO_APPLIED = True


def apply_transforms(
    kg: KnowledgeGraph,
    transforms: Transforms,
    run_config: RunConfig = RunConfig(),
    callbacks: t.Optional[Callbacks] = None,
):
    """
    Apply a list of transformations to a knowledge graph in place,
    using ThreadPoolExecutor for parallelism without pickling issues.
    Numpy and IO ops will run in parallel.
    """
    # apply nest_asyncio to fix the event loop issue in jupyter
    apply_nest_asyncio()

    # if single transformation, wrap it in a list
    if isinstance(transforms, BaseGraphTransformation):
        transforms = [transforms]

    # If running Parallel transforms, use ThreadPoolExecutor instead of ProcessPoolExecutor
    if isinstance(transforms, Parallel):
        transforms_list = transforms.transformations
        max_workers = min(
            len(transforms_list),
            run_config.max_workers if run_config.max_workers > 0 else 4,
        )

        logger.info(
            f"Applying {len(transforms_list)} transforms in parallel with {max_workers} workers"
        )

        def apply_transform(transform):
            """Apply a transformation and return its results to be merged later"""
            logger.info(f"Starting transform: {transform.__class__.__name__}")

            # Create a local copy of the knowledge graph for this transform
            # to avoid thread safety issues with shared data structures
            local_kg = KnowledgeGraph(nodes=kg.nodes.copy(), relationships=[])

            # Run the coroutines for this transform on the local copy
            # TODO: Untested refactor
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    run_coroutines(
                        transform.generate_execution_plan(local_kg),
                        get_desc(transform),
                        1,  # Use a single worker within each thread
                    )
                )
            finally:
                loop.close()

            logger.info(f"Completed transform: {transform.__class__.__name__}")
            return local_kg.relationships

        # Use ThreadPoolExecutor to run transforms in parallel
        # Can run the numpy operations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(apply_transform, transform)
                for transform in transforms_list
            ]

            # Wait for all futures to complete and collect relationships
            for future in concurrent.futures.as_completed(futures):
                try:
                    # Merge relationships back into the original KG
                    relationships = future.result()
                    kg.relationships.extend(relationships)
                except Exception as e:
                    logger.error(f"Error in transformation: {e}")

    # If running sequential transforms, keep the original behavior
    elif isinstance(transforms, t.List):
        for transform in transforms:
            logger.info(f"Applying transform: {transform.__class__.__name__}")
            asyncio.run(
                run_coroutines(
                    transform.generate_execution_plan(kg),
                    get_desc(transform),
                    run_config.max_workers,
                )
            )
    else:
        raise ValueError(
            f"Invalid transforms type: {type(transforms)}. Expects a list of BaseGraphTransformations or a Parallel instance."
        )


def rollback_transforms(kg: KnowledgeGraph, transforms: Transforms):
    """
    Rollback a list of transformations from a knowledge graph.

    Note
    ----
    This is not yet implemented. Please open an issue if you need this feature.
    """
    # this will allow you to roll back the transformations
    raise NotImplementedError
