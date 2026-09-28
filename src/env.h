#pragma once

#include <cstdint>
#include <string>

namespace ctranslate2 {

  std::string read_string_from_env(const char* var, const std::string& default_value = "");
  bool read_bool_from_env(const char* var, const bool default_value = false);
  int read_int_from_env(const char* var, const int default_value = 0);

  // For a feature that is on by default: false only for "0" or "false" (any case), true
  // for anything else, including an unset or empty variable. read_bool_from_env with a
  // true default would instead turn the feature off for "on", "yes" or "True".
  bool parse_default_on_flag(const std::string& value);
  bool read_default_on_flag_from_env(const char* var);

  // Parses a byte count: a non-negative decimal integer with an optional K, M or G suffix
  // (binary multiples, case-insensitive), or "max" for the largest count. Throws
  // std::invalid_argument on anything else, including a count that overflows.
  uint64_t parse_byte_size(const std::string& value);
  uint64_t read_byte_size_from_env(const char* var, const uint64_t default_value = 0);

}
