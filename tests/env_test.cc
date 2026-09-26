#include <limits>
#include <stdexcept>

#include "env.h"
#include "test_utils.h"

TEST(EnvTest, ParseByteSize) {
  // The parser behind CT2_CUDA_POOL_RELEASE_THRESHOLD (read_byte_size_from_env).
  EXPECT_EQ(parse_byte_size("0"), uint64_t(0));
  EXPECT_EQ(parse_byte_size("4096"), uint64_t(4096));
  EXPECT_EQ(parse_byte_size("512K"), uint64_t(512) << 10);
  EXPECT_EQ(parse_byte_size("512k"), uint64_t(512) << 10);
  EXPECT_EQ(parse_byte_size("256M"), uint64_t(256) << 20);
  EXPECT_EQ(parse_byte_size("8G"), uint64_t(8) << 30);
  EXPECT_EQ(parse_byte_size("8g"), uint64_t(8) << 30);

  constexpr uint64_t max_size = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(parse_byte_size("max"), max_size);
  EXPECT_EQ(parse_byte_size("MAX"), max_size);
  EXPECT_EQ(parse_byte_size("18446744073709551615"), max_size);
  EXPECT_EQ(parse_byte_size("17179869183G"), uint64_t(17179869183) << 30);
}

TEST(EnvTest, ParseByteSizeRejectsMalformed) {
  for (const char* value : {"", "-1", "+1", " 1", "1 ", "1.5G", "8GB", "G", "1T", "maxi",
                            // One past the largest count, bare and through the multiplier.
                            "18446744073709551616", "17179869184G"}) {
    EXPECT_THROW(parse_byte_size(value), std::invalid_argument) << "'" << value << "'";
  }
}
