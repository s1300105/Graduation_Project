uint32_t BinaryProtocolWriter::serializedSizeBinary(
    folly::StringPiece str) const {
  return serializedSizeBinary(folly::ByteRange(str));
}