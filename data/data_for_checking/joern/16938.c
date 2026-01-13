Word getHeaderMapSizeHandler(void* raw_context, Word type, Word result_ptr) {
  if (type.u64_ > static_cast<uint64_t>(HeaderMapType::MAX)) {
    return wasmResultToWord(WasmResult::BadArgument);
  }
  auto context = WASM_CONTEXT(raw_context);
  size_t result = context->getHeaderMapSize(static_cast<HeaderMapType>(type.u64_));
  if (!context->wasmVm()->setWord(result_ptr.u64_, Word(result))) {
    return wasmResultToWord(WasmResult::InvalidMemoryAccess);
  }
  return wasmResultToWord(WasmResult::Ok);
}