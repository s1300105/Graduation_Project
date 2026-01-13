inline bool canReadNElements(
    Protocol_& prot,
    uint32_t n,
    std::initializer_list<
        typename detail::ProtocolReaderWireTypeInfo<Protocol_>::WireType>
        types) {
  return prot.getCursor().canAdvance(n * types.size());
}