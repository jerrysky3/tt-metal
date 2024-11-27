// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "common/core_coord.hpp"
#include "dispatch_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "host_api.hpp"
#include "compile_program_with_kernel_path_env_var_fixture.hpp"

using namespace tt;

// Ensures we can successfully create kernels on available compute grid
TEST_F(DispatchFixture, TensixCreateKernelsOnComputeCores) {
    for (unsigned int id = 0; id < this->devices_.size(); id++) {
        tt_metal::Program program = CreateProgram();
        CoreCoord compute_grid = this->devices_.at(id)->compute_with_storage_grid_size();
        EXPECT_NO_THROW(
            auto test_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default}););
    }
}

TEST_F(DispatchFixture, DISABLED_TensixCreateKernelsOnStorageCores) {
    for (unsigned int id = 0; id < this->devices_.size(); id++) {
        if (this->devices_.at(id)->storage_only_cores().empty()) {
            GTEST_SKIP() << "This test only runs on devices with storage only cores";
        }
        tt_metal::Program program = CreateProgram();
        const std::set<CoreCoord>& storage_only_cores = this->devices_.at(id)->storage_only_cores();
        std::set<CoreRange> storage_only_core_ranges;
        for (CoreCoord core : storage_only_cores) {
            storage_only_core_ranges.emplace(core);
        }
        CoreRangeSet storage_core_range_set(storage_only_core_ranges);
        EXPECT_ANY_THROW(
            auto test_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                storage_core_range_set,
                DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default}););
    }
}

TEST_F(DispatchFixture, DISABLED_TensixIdleEthCreateKernelsOnDispatchCores) {
    if (this->IsSlowDispatch()) {
        GTEST_SKIP() << "This test is only supported in fast dispatch mode";
    }
    for (unsigned int id = 0; id < this->devices_.size(); id++) {
        tt_metal::Program program = CreateProgram();
        Device* device = this->devices_.at(id);
        CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
        std::vector<CoreCoord> dispatch_cores = tt::get_logical_dispatch_cores(device->id(), device->num_hw_cqs(), dispatch_core_type);
        std::set<CoreRange> dispatch_core_ranges;
        for (CoreCoord core : dispatch_cores) {
            dispatch_core_ranges.emplace(core);
        }
        CoreRangeSet dispatch_core_range_set(dispatch_core_ranges);
        if (dispatch_core_type == CoreType::WORKER) {
            EXPECT_ANY_THROW(
                auto test_kernel = tt_metal::CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                    CoreRangeSet(dispatch_core_range_set),
                    DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default}););
        } else if (dispatch_core_type == CoreType::ETH) {
            EXPECT_ANY_THROW(auto test_kernel = tt_metal::CreateKernel(
                                 program,
                                 "tests/tt_metal/tt_metal/test_kernels/misc/erisc_print.cpp",
                                 CoreRangeSet(dispatch_core_range_set),
                                 EthernetConfig{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0}););
        }
    }
}

TEST_F(CompileProgramWithKernelPathEnvVarFixture, TensixKernelUnderMetalRootDir) {
    const string &kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
    create_kernel(kernel_file);
    detail::CompileProgram(this->device_, this->program_);
}

TEST_F(CompileProgramWithKernelPathEnvVarFixture, TensixKernelUnderKernelRootDir) {
    const string &orig_kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
    const string &new_kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/new_kernel.cpp";
    this->setup_kernel_dir(orig_kernel_file, new_kernel_file);
    this->create_kernel(new_kernel_file);
    detail::CompileProgram(this->device_, this->program_);
    this->cleanup_kernel_dir();
}

TEST_F(CompileProgramWithKernelPathEnvVarFixture, TensixKernelUnderMetalRootDirAndKernelRootDir) {
    const string &kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
    this->setup_kernel_dir(kernel_file, kernel_file);
    this->create_kernel(kernel_file);
    detail::CompileProgram(this->device_, this->program_);
    this->cleanup_kernel_dir();
}

TEST_F(CompileProgramWithKernelPathEnvVarFixture, TensixNonExistentKernel) {
    const string &kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/non_existent_kernel.cpp";
    this->create_kernel(kernel_file);
    EXPECT_THROW(detail::CompileProgram(this->device_, this->program_), std::exception);
}