TEST(ProtocolTruncatedDataTest, TruncatedSet) {
  TestStruct s;
  s.i32_set_ref() = {};
  for (size_t i = 0; i < 30; ++i) {
    s.i32_set_ref()->emplace((1ull << i));
  }

  testPartialDataHandling<CompactSerializer>(
      s, 3 /* headers */ + 30 /* 1b / element */);
}