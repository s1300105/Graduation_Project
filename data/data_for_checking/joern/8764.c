const CString& CUser::GetAltNick(bool bAllowDefault) const {
    return (bAllowDefault && m_sAltNick.empty()) ? GetCleanUserName()
                                                 : m_sAltNick;
}