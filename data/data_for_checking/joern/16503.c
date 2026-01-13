TYPED_TEST(MultiProtocolTest, EmptyStructWithInclude) {
  EXPECT_COMPATIBLE_PROTOCOL(
      StructWithInclude(), tablebased::StructWithInclude(), TypeParam);
}