WasmResult Context::resolveSharedQueue(absl::string_view vm_id, absl::string_view queue_name,
                                       uint32_t* token_ptr) {
  uint32_t token = global_shared_data.resolveQueue(vm_id, queue_name);
  if (!token) {
    return WasmResult::NotFound;
  }
  *token_ptr = token;
  return WasmResult::Ok;
}