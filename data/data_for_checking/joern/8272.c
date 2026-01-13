CModule::EModRet CModule::OnChanActionMessage(CActionMessage& Message) {
    CString sText = Message.GetText();
    EModRet ret = OnChanAction(Message.GetNick(), *Message.GetChan(), sText);
    Message.SetText(sText);
    return ret;
}