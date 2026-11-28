// SPDX-License-Identifier: BSD-3-Clause
// Shared NSR1 replay protocol for explicitly compiled POSIX HDL adapters.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace spice_cosim {
using Bytes = std::vector<std::uint8_t>;
constexpr std::uint64_t no_tick = std::numeric_limits<std::uint64_t>::max();
constexpr std::array<std::uint8_t, 4> magic{'N', 'S', 'R', '1'};
enum class Operation { apply, advance };

inline std::uint64_t read_le(std::span<const std::uint8_t> &data, std::size_t width) {
    if (data.size() < width) throw std::runtime_error("short request");
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < width; ++i) value |= std::uint64_t(data[i]) << (8*i);
    data = data.subspan(width);
    return value;
}

inline void append_le(Bytes &data, std::uint64_t value, std::size_t width) {
    for (std::size_t i = 0; i < width; ++i)
        data.push_back(std::uint8_t(value >> (8*i)));
}

struct Request {
    std::uint64_t tick;
    Operation operation;
    Bytes inputs;
};

class Wire {
    FILE *file;
    void write_frame(const Bytes &payload) {
        Bytes prefix;
        append_le(prefix, payload.size(), 4);
        if (std::fwrite(prefix.data(), 1, prefix.size(), file) != prefix.size()
            || std::fwrite(payload.data(), 1, payload.size(), file) != payload.size()
            || std::fflush(file) != 0)
            throw std::runtime_error("protocol write failed");
    }
public:
    Wire() {
        const int fd = ::dup(STDOUT_FILENO);
        if (fd < 0) throw std::runtime_error("cannot duplicate protocol output");
        file = ::fdopen(fd, "wb");
        if (!file || ::dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
            if (file) std::fclose(file); else ::close(fd);
            throw std::runtime_error("cannot isolate protocol output");
        }
    }
    ~Wire() { std::fclose(file); }
    Wire(const Wire &) = delete;
    Wire &operator=(const Wire &) = delete;

    std::optional<Request> read(int precision, std::size_t inputs, std::size_t outputs) {
        if (inputs > 65536 || outputs == 0 || outputs > 65536)
            throw std::runtime_error("unsupported port width");
        std::array<std::uint8_t,4> prefix{};
        auto count = std::fread(prefix.data(), 1, prefix.size(), stdin);
        if (count == 0 && std::feof(stdin)) return std::nullopt;
        if (count != prefix.size()) throw std::runtime_error("truncated frame length");
        std::span<const std::uint8_t> prefix_view(prefix);
        const auto size = read_le(prefix_view, 4);
        if (size != 36 + inputs) throw std::runtime_error("invalid request length");
        Bytes frame(size);
        if (std::fread(frame.data(), 1, frame.size(), stdin) != frame.size())
            throw std::runtime_error("truncated frame");
        std::span<const std::uint8_t> data(frame);
        if (!std::equal(magic.begin(), magic.end(), data.begin()))
            throw std::runtime_error("expected NSR1 replay protocol");
        data = data.subspan(magic.size());
        const auto tick = read_le(data, 8);
        const auto exponent = read_le(data, 4);
        const auto operation = read_le(data, 4);
        (void)read_le(data, 4); // SPICE delta; adapter settles native deltas.
        const auto in_count = read_le(data, 4);
        const auto out_count = read_le(data, 4);
        const auto snapshot = read_le(data, 4);
        if (exponent != std::uint32_t(precision))
            throw std::runtime_error("HDL time precision differs from bridge precision");
        if (in_count != inputs || out_count != outputs || data.size() != inputs
            || snapshot != std::numeric_limits<std::uint32_t>::max() || operation > 1
            || std::any_of(data.begin(), data.end(), [](auto bit) { return bit > 3; }))
            throw std::runtime_error("port width, logic or replay mode mismatch");
        return Request{tick, operation == 0 ? Operation::apply : Operation::advance,
                       Bytes(data.begin(), data.end())};
    }

    void reply(std::uint64_t reached, std::uint64_t next, const Bytes &values) {
        Bytes response(magic.begin(), magic.end());
        append_le(response, 0, 4);
        append_le(response, reached, 8);
        append_le(response, next, 8);
        append_le(response, values.size(), 4);
        append_le(response, 0, 4); // Rust owns replay checkpoint.
        response.insert(response.end(), values.begin(), values.end());
        write_frame(response);
    }

    void error(const char *what) {
        Bytes response(magic.begin(), magic.end());
        append_le(response, 1, 4);
        std::string message(what);
        message.resize(std::min<std::size_t>(message.size(), 4096));
        response.insert(response.end(), message.begin(), message.end());
        write_frame(response);
    }
};
}
