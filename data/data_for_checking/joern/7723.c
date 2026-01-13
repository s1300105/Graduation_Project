void CompactProtocolReader::readI16(int16_t& i16) {
  int32_t value;
  apache::thrift::util::readVarint(in_, value);
  i16 = (int16_t)apache::thrift::util::zigzagToI32(value);
}