  uint32_t resolveQueue(absl::string_view vm_id, absl::string_view queue_name) {
    absl::WriterMutexLock l(&mutex);
    auto key = std::make_pair(std::string(vm_id), std::string(queue_name));
    auto it = queue_tokens.find(key);
    if (it != queue_tokens.end()) {
      return it->second;
    }
    return 0; // N.B. zero indicates that the queue was not found.
  }