  static NAN_MODULE_INIT(Init) {
    Local<FunctionTemplate> tpl = Nan::New<FunctionTemplate>(New);
    tpl->SetClassName(Nan::New("ChaChaPolyCipher").ToLocalChecked());
    tpl->InstanceTemplate()->SetInternalFieldCount(1);

    SetPrototypeMethod(tpl, "encrypt", Encrypt);
    SetPrototypeMethod(tpl, "free", Free);

    constructor().Reset(Nan::GetFunction(tpl).ToLocalChecked());

    Nan::Set(target,
             Nan::New("ChaChaPolyCipher").ToLocalChecked(),
             Nan::GetFunction(tpl).ToLocalChecked());
  }