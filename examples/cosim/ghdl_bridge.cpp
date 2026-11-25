// SPDX-License-Identifier: BSD-3-Clause
// GHDL VPI adapter. Uses public callbacks rather than private runtime symbols.
// A horizon request stops at the first settled output change or its deadline.
#include <vpi_user.h>
#include "wire.hpp"
#include <cstdlib>
#include <memory>
#include <string_view>
#include <utility>

namespace {
using spice_cosim::Bytes;
using spice_cosim::Operation;
using spice_cosim::Request;

std::uint64_t now() {
    s_vpi_time time{};
    time.type = vpiSimTime;
    vpi_get_time(nullptr, &time);
    return (std::uint64_t(time.high) << 32) | time.low;
}

vpiHandle callback(int reason, PLI_INT32 (*fn)(p_cb_data),
                   std::uint64_t delay = 0, vpiHandle object = nullptr) {
    s_vpi_time time{};
    time.type = vpiSimTime;
    time.high = std::uint32_t(delay >> 32);
    time.low = std::uint32_t(delay);
    s_cb_data cb{};
    cb.reason = reason;
    cb.cb_rtn = fn;
    cb.obj = object;
    cb.time = &time;
    auto handle = vpi_register_cb(&cb);
    if (!handle) throw std::runtime_error("GHDL rejected VPI callback");
    return handle;
}

struct Port {
    vpiHandle handle;
    std::size_t width;

    void write(std::span<const std::uint8_t> bits) const {
        std::string chars;
        chars.reserve(width);
        for (auto bit : bits) chars.push_back("01XZ"[bit]);
        s_vpi_value value{};
        value.format = vpiBinStrVal;
        value.value.str = chars.data();
        vpi_put_value(handle, &value, nullptr, vpiNoDelay);
    }

    void read(Bytes &bits) const {
        s_vpi_value value{};
        value.format = vpiBinStrVal;
        vpi_get_value(handle, &value);
        if (!value.value.str) throw std::runtime_error("GHDL returned no port value");
        const std::string_view chars(value.value.str);
        if (chars.size() != width) throw std::runtime_error("GHDL port value width mismatch");
        for (const char c : chars) {
            switch (c) {
            case '0': case 'L': bits.push_back(0); break;
            case '1': case 'H': bits.push_back(1); break;
            case 'U': case 'X': case 'W': case '-': bits.push_back(2); break;
            case 'Z': bits.push_back(3); break;
            default: throw std::runtime_error("unsupported GHDL logic value");
            }
        }
    }
};

class Bridge {
    spice_cosim::Wire wire;
    std::vector<Port> inputs, outputs;
    std::size_t input_width = 0, output_width = 0;
    int precision = 0;
    Bytes held, last;
    std::optional<Request> pending;
    vpiHandle timer = nullptr;
    bool settling = false;
    bool initialized = false;

    static PLI_INT32 start_cb(p_cb_data);
    static PLI_INT32 timer_cb(p_cb_data);
    static PLI_INT32 settled_cb(p_cb_data);
    static PLI_INT32 changed_cb(p_cb_data);

    void discover_ports() {
        precision = vpi_get(vpiTimePrecision, nullptr);
        auto modules = vpi_iterate(vpiModule, nullptr);
        if (!modules) throw std::runtime_error("no GHDL top-level entity");
        auto top = vpi_scan(modules);
        if (!top || vpi_scan(modules))
            throw std::runtime_error("expected one GHDL top-level entity");
        auto ports = vpi_iterate(vpiPort, top);
        if (!ports) throw std::runtime_error("no GHDL top-level ports");
        while (auto handle = vpi_scan(ports)) {
            const int width = vpi_get(vpiSize, handle);
            if (width <= 0 || width > 65536)
                throw std::runtime_error("unsupported GHDL port width");
            Port port{handle, std::size_t(width)};
            switch (vpi_get(vpiDirection, handle)) {
            case vpiInput:
                input_width += port.width;
                inputs.push_back(port);
                break;
            case vpiOutput:
                output_width += port.width;
                outputs.push_back(port);
                break;
            default:
                throw std::runtime_error("GHDL inout or unsupported port direction");
            }
        }
        if (input_width > 65536 || output_width == 0 || output_width > 65536)
            throw std::runtime_error("unsupported flattened GHDL port width");
        for (const auto &port : outputs)
            callback(cbValueChange, changed_cb, 0, port.handle);
    }

    void apply_inputs(const Bytes &bits) {
        std::size_t offset = 0;
        for (const auto &port : inputs) {
            auto slice = std::span<const std::uint8_t>(bits).subspan(offset, port.width);
            if (!initialized || !std::equal(slice.begin(), slice.end(), held.begin()+offset))
                port.write(slice);
            offset += port.width;
        }
        held = bits;
    }

    Bytes read_outputs() const {
        Bytes values;
        values.reserve(output_width);
        for (const auto &port : outputs) port.read(values);
        return values;
    }

    void request_settle() {
        if (!settling) {
            settling = true;
            callback(cbReadWriteSynch, settled_cb);
        }
    }

    void start() {
        discover_ports();
        pending = wire.read(precision, input_width, output_width);
        if (!pending) { vpi_control(vpiFinish, 0); return; }
        if (pending->tick != 0 || pending->operation != Operation::apply)
            throw std::runtime_error("first GHDL request must apply inputs at time zero");
        apply_inputs(pending->inputs);
        request_settle();
    }

    // We are paused at a settled time boundary. Fast-path equal-time requests
    // with unchanged inputs. A same-time input change schedules another delta.
    void receive() {
        for (;;) {
            pending = wire.read(precision, input_width, output_width);
            if (!pending) { vpi_control(vpiFinish, 0); return; }
            const auto current = now();
            if (pending->tick < current)
                throw std::runtime_error("backwards GHDL time");
            if (pending->operation == Operation::advance && pending->inputs != held)
                throw std::runtime_error("lookahead changed held input values");
            if (pending->tick == current && pending->inputs == held) {
                wire.reply(current, spice_cosim::no_tick, last);
                continue;
            }
            timer = callback(cbAfterDelay, timer_cb, pending->tick-current);
            return;
        }
    }

    void deadline() {
        timer = nullptr; // GHDL owns and retires a fired callback handle.
        if (!pending || now() != pending->tick)
            throw std::runtime_error("unexpected GHDL timer");
        if (pending->operation == Operation::apply) apply_inputs(pending->inputs);
        request_settle();
    }

    void settled() {
        settling = false;
        if (!pending) return;
        const auto current = now();
        auto values = read_outputs();
        if (current < pending->tick) {
            if (values == last) return; // transient delta glitch, not a settled event.
            if (pending->operation == Operation::apply)
                throw std::runtime_error("input request skipped an earlier GHDL output");
            if (timer && !vpi_remove_cb(timer))
                throw std::runtime_error("cannot cancel GHDL horizon timer");
            timer = nullptr;
        } else if (current != pending->tick) {
            throw std::runtime_error("GHDL overshot requested horizon");
        }
        initialized = true;
        last = std::move(values);
        wire.reply(current, spice_cosim::no_tick, last);
        receive();
    }

public:
    void register_start() { callback(cbStartOfSimulation, start_cb); }

    [[noreturn]] void fail(const char *error) {
        try { wire.error(error); } catch (...) {}
        std::exit(1);
    }
};

std::unique_ptr<Bridge> bridge;

template<void (Bridge::*Fn)()>
PLI_INT32 dispatch(p_cb_data) {
    try { ((*bridge).*Fn)(); }
    catch (const std::exception &error) { bridge->fail(error.what()); }
    return 0;
}

PLI_INT32 Bridge::start_cb(p_cb_data cb) { return dispatch<&Bridge::start>(cb); }
PLI_INT32 Bridge::timer_cb(p_cb_data cb) { return dispatch<&Bridge::deadline>(cb); }
PLI_INT32 Bridge::settled_cb(p_cb_data cb) { return dispatch<&Bridge::settled>(cb); }
PLI_INT32 Bridge::changed_cb(p_cb_data cb) { return dispatch<&Bridge::request_settle>(cb); }

void startup() {
    try {
        bridge = std::make_unique<Bridge>();
        bridge->register_start();
    } catch (const std::exception &error) {
        if (bridge) bridge->fail(error.what());
        std::exit(1);
    }
}
}

extern "C" {
void (*vlog_startup_routines[])() = {startup, nullptr};
}
