Http::FilterDataStatus Context::decodeData(Buffer::Instance& data, bool end_stream) {
  requestBodyBuffer_ = &data;
  auto result = onRequestBody(data.length(), end_stream);
  requestBodyBuffer_ = nullptr;
  return result;
}