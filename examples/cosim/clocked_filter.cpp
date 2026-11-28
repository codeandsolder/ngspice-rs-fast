// SPDX-License-Identifier: BSD-3-Clause
#include "Vclocked_filter.h"
#include "verilator_bridge.hpp"

int main() {
    return spice_cosim::run<Vclocked_filter>(1, 2,
        [](auto &model, auto inputs) { model.sample = inputs[0]; },
        [](const auto &model) {
            return spice_cosim::Bytes{std::uint8_t(model.filtered), std::uint8_t(model.clock)};
        });
}
