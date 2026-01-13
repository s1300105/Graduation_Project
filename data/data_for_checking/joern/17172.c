uint32_t CompactProtocolWriter::writeSetBegin(TType elemType, uint32_t size) {
  return writeCollectionBegin(elemType, size);
}