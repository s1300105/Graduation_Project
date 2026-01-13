void CModule::OnRawMode2(const CNick* pOpNick, CChan& Channel,
                         const CString& sModes, const CString& sArgs) {
    if (pOpNick) OnRawMode(*pOpNick, Channel, sModes, sArgs);
}