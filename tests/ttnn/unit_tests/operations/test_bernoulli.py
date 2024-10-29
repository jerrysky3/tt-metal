# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import numpy as np
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    get_lib_dtype,
)
from models.utility_functions import skip_for_grayskull
from collections import Counter
from loguru import logger


def run_bernoulli(shape, in_dtype, out_dtype, device, is_out_alloc=False, compute_kernel_options=None):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    cpu_input = torch.rand(shape, dtype=get_lib_dtype(torch, in_dtype))
    npu_input = ttnn.from_torch(cpu_input, device=device, dtype=get_lib_dtype(ttnn, in_dtype), layout=ttnn.TILE_LAYOUT)

    npu_output = None
    if is_out_alloc:
        cpu_output = torch.rand(shape, dtype=get_lib_dtype(torch, out_dtype))
        npu_output = ttnn.from_torch(
            cpu_output, device=device, dtype=get_lib_dtype(ttnn, out_dtype), layout=ttnn.TILE_LAYOUT
        )

    one_probs = []
    for _ in range(10):
        if is_out_alloc:
            ttnn.bernoulli(
                npu_input,
                output=npu_output,
                dtype=get_lib_dtype(ttnn, out_dtype),
                compute_kernel_config=compute_kernel_config,
            )
        else:
            npu_output = ttnn.bernoulli(
                npu_input,
                dtype=get_lib_dtype(ttnn, out_dtype),
                compute_kernel_config=compute_kernel_config,
            )

        tt_output = ttnn.to_torch(npu_output).reshape(shape)
        tt_output_list = tt_output.flatten().tolist()

        c = Counter(tt_output_list)
        one_probs.append(c[1] / len(tt_output_list))

    expected_one_prob = 0.5
    assert np.allclose(expected_one_prob, np.mean(one_probs), rtol=0.05)


# fmt: off
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("shape",
    [
        [2003],
        [500, 500],
        [1, 512, 2, 256],
    ],
)
@pytest.mark.parametrize("in_dtype",
    [
        "bfloat16",
        "float32"
    ]
)
@pytest.mark.parametrize("out_dtype",
    [
        "bfloat16",
        "float32"
    ]
)
@pytest.mark.parametrize("is_out_alloc",
    [
        True,
        False
    ]
)
# fmt: on
def test_bernoulli(shape, in_dtype, out_dtype, device, is_out_alloc):
    torch.manual_seed(0)
    run_bernoulli(shape, in_dtype, out_dtype, device, is_out_alloc)


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "shape",
    [
        [1, 21, 123, 24],
    ],
)
@pytest.mark.parametrize("in_dtype", ["float32"])
@pytest.mark.parametrize("out_dtype", ["float32"])
@pytest.mark.parametrize("is_out_alloc", [True, False])
def test_bernoulli_callback(shape, in_dtype, out_dtype, device, is_out_alloc, use_program_cache):
    torch.manual_seed(0)
    num_program_cache_entries_list = []
    for i in range(2):
        run_bernoulli(shape, in_dtype, out_dtype, device, is_out_alloc)
        # Add dummy tensor to make sure that created tensor in 2 iteration don't share the same addr
        tt_dummy_tensor = ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "shape",
    [[512, 512], [5, 4, 70, 40]],
)
@pytest.mark.parametrize("in_dtype", ["float32"])
@pytest.mark.parametrize("out_dtype", ["float32"])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_uniform_with_compute_kernel_options(shape, in_dtype, out_dtype, device, compute_kernel_options):
    torch.manual_seed(0)
    run_bernoulli(shape, in_dtype, out_dtype, device, compute_kernel_options)