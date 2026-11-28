// SPDX-License-Identifier: BSD-3-Clause
#include "Vfeedback_inverter.h"
#include "verilator_bridge.hpp"

int main() {
    return spice_cosim::run<Vfeedback_inverter>(1, 1,
        [](auto &model, auto inputs) { model.sense = inputs[0]; },
        [](const auto &model) { return spice_cosim::Bytes{std::uint8_t(model.control)}; });
}
