TYPED_TEST(MultiProtocolTest, FrozenStructB) {
  FrozenStructB oldObject = makeFrozenStructBLike<FrozenStructB>();
  tablebased::FrozenStructB newObject =
      makeFrozenStructBLike<tablebased::FrozenStructB>();
  EXPECT_COMPATIBLE_PROTOCOL(oldObject, newObject, TypeParam);
}