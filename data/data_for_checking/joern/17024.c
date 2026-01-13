uint32_t CompactProtocolWriter::serializedSizeMapBegin(
    TType /*keyType*/,
    TType /*valType*/,
    uint32_t /*size*/) const {
  return serializedSizeByte() + serializedSizeByte() + serializedSizeI32();
}