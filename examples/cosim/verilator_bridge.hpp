// SPDX-License-Identifier: BSD-3-Clause
// Forward-only HDL bridge for timed Verilator models.
#pragma once
#include <verilated.h>
#include "wire.hpp"

namespace spice_cosim {
// Input/output mapping is explicit, in ordered flattened bits. Both callbacks
// operate on the compiled top-level model and must match the declared widths.
template<class Model, class SetInputs, class GetOutputs>
int run(std::size_t inputs, std::size_t outputs, SetInputs set, GetOutputs get) {
    Wire wire;
    try {
        VerilatedContext context;
        context.threads(1);
        context.randReset(0);
        context.randSeed(1);
        Model model(&context);
        bool started = false;
        while (auto request = wire.read(context.timeprecision(), inputs, outputs)) {
            if (request->operation != Operation::apply)
                throw std::runtime_error("Verilator adapter reports direct wakeups");
            if (std::any_of(request->inputs.begin(), request->inputs.end(), [](auto bit) { return bit > 1; }))
                throw std::runtime_error("Verilator is two-state: X/Z input is not representable");
            if (request->tick < context.time()
                || (!started && request->tick != 0)
                || (started && model.eventsPending() && model.nextTimeSlot() < request->tick))
                throw std::runtime_error("missed or backwards HDL time slot");
            context.time(request->tick);
            set(model, std::span<const std::uint8_t>(request->inputs));
            model.eval();
            started = true;
            if (context.gotFinish()) throw std::runtime_error("HDL simulation finished");
            auto next = model.eventsPending() ? model.nextTimeSlot() : no_tick;
            Bytes values = get(model);
            if (values.size() != outputs
                || std::any_of(values.begin(), values.end(), [](auto bit) { return bit > 1; }))
                throw std::runtime_error("invalid two-state output mapping");
            wire.reply(request->tick, next, values);
        }
        model.final();
    } catch (const std::exception &error) {
        try { wire.error(error.what()); } catch (...) {}
        return 1;
    }
    return 0;
}
}
