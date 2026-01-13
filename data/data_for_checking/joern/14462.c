TYPED_TEST(MultiProtocolTest, EmptyStructA) {
  EXPECT_COMPATIBLE_PROTOCOL(StructA(), tablebased::StructA(), TypeParam);
}