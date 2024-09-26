// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {
    constexpr CoreCoord core = {0, 0};
    int device_id = 0;
    Device *device = CreateDevice(device_id);
    CommandQueue &cq = device->command_queue();
    Program program = CreateProgram();

    KernelHandle read_dataflow_kernel_noc0_id = CreateKernel(
        program,
        "tt_metal/programming_examples/pow_test/kernels/dataflow/read_dataflow_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle write_dataflow_kernel_noc1_id = CreateKernel(
        program,
        "tt_metal/programming_examples/pow_test/kernels/dataflow/write_dataflow_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    vector<uint32_t> compute_kernel_args = {};
    KernelHandle test_compute_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/pow_test/kernels/compute/test_compute_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args});

    constexpr uint32_t single_tile_size = 2 * (32 * 32);
    constexpr uint32_t num_tiles = 64;
    constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    auto input_dram_buffer = CreateBuffer(dram_config);
    auto output_dram_buffer = CreateBuffer(dram_config);

    constexpr uint32_t src0_cb_index = CB::c_in0;
    constexpr uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t output_cb_index = CB::c_out0;
    constexpr uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, single_tile_size);
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(dram_buffer_size, 100, 573);
    EnqueueWriteBuffer(cq, input_dram_buffer, input_vec, false);

    const std::vector<uint32_t> noc0_args = {
        input_dram_buffer->address(),
        static_cast<uint32_t>(input_dram_buffer->noc_coordinates().x),
        static_cast<uint32_t>(input_dram_buffer->noc_coordinates().y),
        num_tiles,
    };
    SetRuntimeArgs(program, read_dataflow_kernel_noc0_id, core, noc0_args);

    const std::vector<uint32_t> noc1_args = {
        output_dram_buffer->address(),
        static_cast<uint32_t>(output_dram_buffer->noc_coordinates().x),
        static_cast<uint32_t>(output_dram_buffer->noc_coordinates().y),
        num_tiles,
    };
    SetRuntimeArgs(program, write_dataflow_kernel_noc1_id, core, noc1_args);

    const std::vector<uint32_t> compute_args = {
        num_tiles,
    };
    SetRuntimeArgs(program, test_compute_kernel_id, core, compute_args);
    EnqueueProgram(cq, program, false);
    Finish(cq);

    std::vector<uint32_t> output_vec;
    EnqueueReadBuffer(cq, output_dram_buffer, output_vec, true);

    for (size_t i = 0; i < input_vec.size(); ++i) {
        if (input_vec[i] != output_vec[i]) {
            printf("Diff %lu %08x %08x\n", i, input_vec[i], output_vec[i]);
            break;
        }
    }
    printf("Done\n");

    CloseDevice(device);

    return 0;
}
