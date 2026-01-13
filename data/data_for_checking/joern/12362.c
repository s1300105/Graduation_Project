bool Context::validateConfiguration(absl::string_view configuration) {
  if (!wasm_->validateConfiguration_) {
    return true;
  }
  auto address = wasm_->copyString(configuration);
  return wasm_->validateConfiguration_(this, id_, address, configuration.size()).u64_ != 0;
}