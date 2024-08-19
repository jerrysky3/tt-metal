// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "common/core_coord.h"
#include "impl/buffers/buffer.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include "ttnn/run_operation.hpp"

#include <optional>
#include <vector>
#include <algorithm>

/* Fusion includes */
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/cpp/ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/experimental/ccl/ccl_op_fusion.hpp"


namespace ttnn {
namespace experimental {

struct AllGatherMatmul {

    /* All Gather Params */
    const ttnn::AllGather all_gather_struct;

    /* Matmul Params */
    const operations::matmul::Matmul matmul_struct;

    /* Fusion Params */
    const CoreCoord all_gather_core_grid_offset;

    /* General */
    void validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
};

operation::ProgramWithCallbacks all_gather_matmul_multi_core_with_workers(

    /* General Params */
    const Tensor& input_tensor,
    Tensor& all_gather_output_tensor,
    Tensor& matmul_output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    all_gather_op::Topology topology,
    const CoreCoord core_grid_offset = CoreCoord(0, 0)

    /* Matmul Params */
    // const std::optional<const Tensor> bias,
    // Tensor &mm_output_tensor,
    // bool bcast_batch,
    // CoreCoord compute_with_storage_grid_size,
    // DeviceComputeKernelConfig compute_kernel_config,
    // uint32_t in0_block_w,
    // uint32_t out_subblock_h,
    // uint32_t out_subblock_w,
    // uint32_t per_core_M,
    // uint32_t per_core_N,
    // bool fuse_batch,
    // bool transpose_mcast,
    // std::optional<UnaryWithParam> fused_activation,
    // bool untilize_out

);
}  // namespace experimental


namespace operations {
namespace experimental {
namespace ccl {

std::vector<Tensor> all_gather_matmul(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const uint32_t dim,
    const CoreCoord all_gather_core_grid_offset,
    const uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const bool transpose_a = false,
    const bool transpose_b = false,
    const std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const operations::matmul::MatmulProgramConfig> program_config = std::nullopt,
    const std::optional<const std::string>& activation = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<const ttnn::CoreGrid> core_grid = std::nullopt);

} // namespace ccl
} // namespace experimental
} // namespace operations

}  // namespace ttnn