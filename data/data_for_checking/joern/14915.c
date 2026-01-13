  bool isGaugeMetricId(uint32_t metric_id) {
    return (metric_id & kMetricTypeMask) == kMetricTypeGauge;
  }