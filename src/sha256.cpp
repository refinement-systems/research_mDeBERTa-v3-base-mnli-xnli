#include "sha256.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {

constexpr std::array<uint32_t, 64> kSha256Constants = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

inline uint32_t RotateRight(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32u - bits));
}

struct Sha256State {
    std::array<uint32_t, 8> hash = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    std::array<uint8_t, 64> buffer = {};
    size_t buffer_size = 0;
    uint64_t total_bits = 0;
};

void TransformBlock(Sha256State& state, const uint8_t block[64]) {
    std::array<uint32_t, 64> schedule = {};
    for (size_t i = 0; i < 16; ++i) {
        const size_t offset = i * 4;
        schedule[i] =
            (static_cast<uint32_t>(block[offset]) << 24u) |
            (static_cast<uint32_t>(block[offset + 1]) << 16u) |
            (static_cast<uint32_t>(block[offset + 2]) << 8u) |
            static_cast<uint32_t>(block[offset + 3]);
    }

    for (size_t i = 16; i < schedule.size(); ++i) {
        const uint32_t s0 =
            RotateRight(schedule[i - 15], 7u) ^
            RotateRight(schedule[i - 15], 18u) ^
            (schedule[i - 15] >> 3u);
        const uint32_t s1 =
            RotateRight(schedule[i - 2], 17u) ^
            RotateRight(schedule[i - 2], 19u) ^
            (schedule[i - 2] >> 10u);
        schedule[i] = schedule[i - 16] + s0 + schedule[i - 7] + s1;
    }

    uint32_t a = state.hash[0];
    uint32_t b = state.hash[1];
    uint32_t c = state.hash[2];
    uint32_t d = state.hash[3];
    uint32_t e = state.hash[4];
    uint32_t f = state.hash[5];
    uint32_t g = state.hash[6];
    uint32_t h = state.hash[7];

    for (size_t i = 0; i < schedule.size(); ++i) {
        const uint32_t s1 = RotateRight(e, 6u) ^ RotateRight(e, 11u) ^ RotateRight(e, 25u);
        const uint32_t choice = (e & f) ^ (~e & g);
        const uint32_t temp1 = h + s1 + choice + kSha256Constants[i] + schedule[i];
        const uint32_t s0 = RotateRight(a, 2u) ^ RotateRight(a, 13u) ^ RotateRight(a, 22u);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = s0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state.hash[0] += a;
    state.hash[1] += b;
    state.hash[2] += c;
    state.hash[3] += d;
    state.hash[4] += e;
    state.hash[5] += f;
    state.hash[6] += g;
    state.hash[7] += h;
}

void UpdateSha256(Sha256State& state, const uint8_t* data, size_t size) {
    while (size > 0) {
        const size_t remaining = state.buffer.size() - state.buffer_size;
        const size_t chunk = std::min(remaining, size);
        std::memcpy(state.buffer.data() + state.buffer_size, data, chunk);
        state.buffer_size += chunk;
        data += chunk;
        size -= chunk;

        if (state.buffer_size == state.buffer.size()) {
            TransformBlock(state, state.buffer.data());
            state.total_bits += 512u;
            state.buffer_size = 0;
        }
    }
}

std::string FinalizeSha256(Sha256State& state) {
    const uint64_t total_bits = state.total_bits + (static_cast<uint64_t>(state.buffer_size) * 8u);

    state.buffer[state.buffer_size++] = 0x80u;
    if (state.buffer_size > 56u) {
        while (state.buffer_size < state.buffer.size()) {
            state.buffer[state.buffer_size++] = 0u;
        }
        TransformBlock(state, state.buffer.data());
        state.buffer_size = 0;
    }

    while (state.buffer_size < 56u) {
        state.buffer[state.buffer_size++] = 0u;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        state.buffer[state.buffer_size++] = static_cast<uint8_t>((total_bits >> shift) & 0xffu);
    }
    TransformBlock(state, state.buffer.data());

    std::ostringstream hex;
    hex << std::hex << std::setfill('0');
    for (const uint32_t value : state.hash) {
        hex << std::setw(8) << value;
    }
    return hex.str();
}

}  // namespace

namespace nli {

std::string ComputeFileSha256Hex(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open file for SHA-256: " + path.string());
    }

    Sha256State state;
    std::array<char, 8192> buffer = {};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            UpdateSha256(
                state,
                reinterpret_cast<const uint8_t*>(buffer.data()),
                static_cast<size_t>(count));
        }
    }

    return FinalizeSha256(state);
}

}  // namespace nli

