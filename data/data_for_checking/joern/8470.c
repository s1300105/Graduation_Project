long FS_ReadFile( const char *qpath, void **buffer ) {
	fileHandle_t	h;
	byte*			buf;
	long				len;

	FS_AssertInitialised();

	if ( !qpath || !qpath[0] ) {
		Com_Error( ERR_FATAL, "FS_ReadFile with empty name\n" );
	}

	// stop sounds from repeating
	S_ClearSoundBuffer();

	buf = NULL;	// quiet compiler warning

	// look for it in the filesystem or pack files
	len = FS_FOpenFileRead( qpath, &h, qfalse );
	if ( h == 0 ) {
		if ( buffer ) {
			*buffer = NULL;
		}
		return -1;
	}

	if ( !buffer ) {
		FS_FCloseFile( h);
		return len;
	}

	fs_loadCount++;

	buf = (byte*)Z_Malloc( len+1, TAG_FILESYS, qfalse);
	buf[len]='\0';	// because we're not calling Z_Malloc with optional trailing 'bZeroIt' bool
	*buffer = buf;

	Z_Label(buf, qpath);

	// PRECACE CHECKER!
#ifndef FINAL_BUILD
	if (com_sv_running && com_sv_running->integer && cls.state >= CA_ACTIVE) {	//com_cl_running
		if (strncmp(qpath,"menu/",5) ) {
			Com_DPrintf( S_COLOR_MAGENTA"FS_ReadFile: %s NOT PRECACHED!\n", qpath );
		}
	}
#endif

	FS_Read (buf, len, h);

	// guarantee that it will have a trailing 0 for string operations
	buf[len] = 0;
	FS_FCloseFile( h );
	return len;
}