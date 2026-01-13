  static NAN_METHOD(DecryptLen) {
    MarkPopErrorOnReturn mark_pop_error_on_return;

    ChaChaPolyDecipher* obj =
      ObjectWrap::Unwrap<ChaChaPolyDecipher>(info.Holder());

    if (!Buffer::HasInstance(info[0]) || Buffer::Length(info[0]) != 4)
      return Nan::ThrowTypeError("Missing/Invalid length bytes");

    if (!info[1]->IsUint32())
      return Nan::ThrowTypeError("Missing/Invalid sequence number");

    unsigned char* length_bytes =
      reinterpret_cast<unsigned char*>(Buffer::Data(info[0]));

    uint32_t dec_packet_length;
    ErrorType r = obj->decrypt_length(
      length_bytes,
      Buffer::Length(info[0]),
      Nan::To<uint32_t>(info[1]).FromJust(),
      &dec_packet_length
    );

    switch (r) {
      case kErrNone:
        return info.GetReturnValue().Set(dec_packet_length);
      case kErrPartialDecrypt:
        return Nan::ThrowError("Failed to completely decrypt packet length");
      case kErrOpenSSL: {
        char msg_buf[128] = {0};
        ERR_error_string_n(ERR_get_error(), msg_buf, sizeof(msg_buf));
        return Nan::ThrowError(msg_buf);
      }
      default:
        return Nan::ThrowError("Unknown decrypt failure");
    }
  }