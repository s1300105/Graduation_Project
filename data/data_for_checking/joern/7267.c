WasmResult Context::incrementMetric(uint32_t metric_id, int64_t offset) {
  auto type = static_cast<MetricType>(metric_id & Wasm::kMetricTypeMask);
  if (type == MetricType::Counter) {
    auto it = wasm_->counters_.find(metric_id);
    if (it != wasm_->counters_.end()) {
      if (offset > 0) {
        it->second->add(offset);
        return WasmResult::Ok;
      } else {
        return WasmResult::BadArgument;
      }
      return WasmResult::NotFound;
    }
  } else if (type == MetricType::Gauge) {
    auto it = wasm_->gauges_.find(metric_id);
    if (it != wasm_->gauges_.end()) {
      if (offset > 0) {
        it->second->add(offset);
        return WasmResult::Ok;
      } else {
        it->second->sub(-offset);
        return WasmResult::Ok;
      }
    }
    return WasmResult::NotFound;
  }
  return WasmResult::BadArgument;
}