bool CZNC::ParseConfig(const CString& sConfig, CString& sError) {
    m_sConfigFile = ExpandConfigPath(sConfig, false);

    CConfig config;
    if (!ReadConfig(config, sError)) return false;

    if (!LoadGlobal(config, sError)) return false;

    if (!LoadUsers(config, sError)) return false;

    return true;
}