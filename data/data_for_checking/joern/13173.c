CModule::EModRet CModule::OnUserNoticeMessage(CNoticeMessage& Message) {
    CString sTarget = Message.GetTarget();
    CString sText = Message.GetText();
    EModRet ret = OnUserNotice(sTarget, sText);
    Message.SetTarget(sTarget);
    Message.SetText(sText);
    return ret;
}