#include "env.h"

#include <cctype>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include "ctranslate2/utils.h"

namespace ctranslate2 {

  std::string read_string_from_env(const char* var, const std::string& default_value) {
    const char* value = std::getenv(var);
    if (!value)
      return default_value;
    return value;
  }

  bool read_bool_from_env(const char* var, const bool default_value) {
    return string_to_bool(read_string_from_env(var, default_value ? "1" : "0"));
  }

  int read_int_from_env(const char* var, const int default_value) {
    const std::string value = read_string_from_env(var);
    if (value.empty())
      return default_value;
    return std::stoi(value);
  }

  uint64_t parse_byte_size(const std::string& value) {
    constexpr uint64_t max_size = std::numeric_limits<uint64_t>::max();
    std::string lower = value;
    for (char& c : lower)
      c = std::tolower(static_cast<unsigned char>(c));
    if (lower == "max")
      return max_size;

    const auto invalid = [&value]() {
      return std::invalid_argument("Invalid byte size '" + value + "': expected a "
                                   "non-negative integer with an optional K, M or G "
                                   "suffix, or 'max'");
    };

    // Digits only: std::stoull would accept (and wrap) a leading minus sign.
    size_t end = 0;
    uint64_t count = 0;
    for (; end < value.size() && std::isdigit(static_cast<unsigned char>(value[end])); ++end) {
      const uint64_t digit = value[end] - '0';
      if (count > (max_size - digit) / 10)
        throw invalid();
      count = count * 10 + digit;
    }
    if (end == 0 || value.size() - end > 1)
      throw invalid();

    uint64_t multiplier = 1;
    if (end < value.size()) {
      switch (std::toupper(static_cast<unsigned char>(value[end]))) {
      case 'K':
        multiplier = uint64_t(1) << 10;
        break;
      case 'M':
        multiplier = uint64_t(1) << 20;
        break;
      case 'G':
        multiplier = uint64_t(1) << 30;
        break;
      default:
        throw invalid();
      }
    }

    if (count > max_size / multiplier)
      throw invalid();
    return count * multiplier;
  }

  uint64_t read_byte_size_from_env(const char* var, const uint64_t default_value) {
    const std::string value = read_string_from_env(var);
    if (value.empty())
      return default_value;
    return parse_byte_size(value);
  }

}
