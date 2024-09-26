// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reg_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

namespace NAMESPACE {

void MAIN {
    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_out0 = 16;

    const uint32_t num_of_blocks = get_arg_val<uint32_t>(0);

    init_sfpu(cb_id_in0);

    for (uint32_t idx = 0; idx < num_of_blocks; ++idx) {
        cb_wait_front(cb_id_in0, 1);

        tile_regs_acquire();

        copy_tile(cb_id_in0, 0, 0);

        tile_regs_commit();

        cb_pop_front(cb_id_in0, 1);

        tile_regs_wait();

        cb_reserve_back(cb_id_out0, 1);
        pack_tile(0, cb_id_out0);
        cb_push_back(cb_id_out0, 1);

        tile_regs_release();
    }

    DPRINT_MATH(DPRINT << "Hello, Master, I am running a test compute kernel." << ENDL());
}

}  // namespace NAMESPACE
