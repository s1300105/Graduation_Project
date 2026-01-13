inline uint64_t Wasm::copyString(absl::string_view s) {
  if (s.empty()) {
    return 0; // nullptr
  }
  uint64_t pointer;
  uint8_t* m = static_cast<uint8_t*>(allocMemory((s.size() + 1), &pointer));
  memcpy(m, s.data(), s.size());
  m[s.size()] = 0;
  return pointer;
}