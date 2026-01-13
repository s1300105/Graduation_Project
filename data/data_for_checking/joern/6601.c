CModule::EModRet CModule::OnPrivTextMessage(CTextMessage& Message) {
    CString sText = Message.GetText();
    EModRet ret = OnPrivMsg(Message.GetNick(), sText);
    Message.SetText(sText);
    return ret;
}