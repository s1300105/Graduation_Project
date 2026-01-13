char *FS_BuildOSPath( const char *qpath ) {
	char	temp[MAX_OSPATH];
	static char ospath[4][MAX_OSPATH];
	static int toggle;

	int nextToggle = (toggle + 1)&3;	// allows four returns without clash (increased from 2 during fs_copyfiles 2 enhancement)
	toggle = nextToggle;

	// Fix for filenames that are given to FS with a leading "/" (/botfiles/Foo)
	if (qpath[0] == '\\' || qpath[0] == '/')
		qpath++;

	Com_sprintf( temp, sizeof(temp), "/base/%s", qpath ); // FIXME SP used fs_gamedir here as well (not sure if this func is even used)
	FS_ReplaceSeparators( temp );
	Com_sprintf( ospath[toggle], sizeof( ospath[0] ), "%s%s", fs_basepath->string, temp );

	return ospath[toggle];
}