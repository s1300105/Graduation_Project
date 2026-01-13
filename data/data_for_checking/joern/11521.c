uint32_t BinaryProtocolWriter::writeI64(int64_t i64) {
  out_.writeBE(i64);
  return sizeof(i64);
}